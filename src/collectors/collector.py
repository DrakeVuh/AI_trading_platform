# src/collectors/collector.py
import argparse
import os
import sys
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import pandas as pd
import ccxt
from tqdm import tqdm

# ----------------------------- Logging setup -----------------------------

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "collector.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# ----------------------------- Helpers -----------------------------------

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}

def utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def parse_date(date_str: Optional[str]) -> Optional[int]:
    """
    Accepts 'YYYY-MM-DD' or None. Returns UTC ms or None.
    """
    if not date_str:
        return None
    return utc_ms(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc))

def ensure_outdir(base_out: Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    sym = symbol.replace("/", "")
    outdir = base_out / "raw" / "crypto" / exchange_id / sym / timeframe
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def save_frame(df: pd.DataFrame, outdir: Path, symbol: str, timeframe: str, fmt: str):
    """Append-safe save. Deduplicate by timestamp, sort ascending."""
    sym = symbol.replace("/", "")
    stem = f"ohlcv_{sym}_{timeframe}"
    if fmt == "parquet":
        outpath = outdir / f"{stem}.parquet"
        if outpath.exists():
            old = pd.read_parquet(outpath)
            df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.to_parquet(outpath, index=False)
        logging.info(f"Saved {len(df):,} rows → {outpath}")
    elif fmt == "csv":
        outpath = outdir / f"{stem}.csv"
        if outpath.exists():
            old = pd.read_csv(outpath, parse_dates=["timestamp"])
            df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.to_csv(outpath, index=False)
        logging.info(f"Saved {len(df):,} rows → {outpath}")
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")

def normalize_ohlcv(rows: List[List]) -> pd.DataFrame:
    """CCXT returns [ts, open, high, low, close, volume]"""
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    # Convert to UTC datetime for readability, keep 'ts' for dedupe if needed
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    return df

# ----------------------------- Fetcher ------------------------------------

def init_exchange(exchange_id: str):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")
    klass = getattr(ccxt, exchange_id)
    # Public OHLCV does not need API keys
    exchange = klass({
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })
    return exchange

def fetch_ohlcv_incremental(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: Optional[int],
    to_ms: Optional[int],
    limit: int = 1000,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Incrementally pull OHLCV across [since_ms, to_ms).
    Will respect exchange.rateLimit via enableRateLimit.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
    tf_ms = TIMEFRAME_MS[timeframe]

    # If no 'to_ms' provided, default = now
    if to_ms is None:
        to_ms = exchange.milliseconds()

    # If no 'since_ms' provided, default to last 30 * timeframe candles
    if since_ms is None:
        since_ms = to_ms - (30 * tf_ms)

    all_rows: List[List] = []
    cursor = since_ms
    bar = tqdm(disable=not progress, desc=f"Fetching {symbol} {timeframe}")

    # Protect against infinite loops
    max_iters = 20000

    while cursor < to_ms and max_iters > 0:
        max_iters -= 1
        try:
            # ccxt fetch_ohlcv signature: (symbol, timeframe='1m', since=None, limit=None, params={})
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except ccxt.DDoSProtection as e:
            logging.warning(f"DDoSProtection: sleeping 5s... {e}")
            time.sleep(5)
            continue
        except ccxt.RateLimitExceeded as e:
            sleep_ms = exchange.rateLimit or 1000
            logging.warning(f"RateLimit: sleeping {sleep_ms} ms... {e}")
            time.sleep(sleep_ms / 1000)
            continue
        except ccxt.NetworkError as e:
            logging.warning(f"NetworkError: retry 3s... {e}")
            time.sleep(3)
            continue
        except Exception as e:
            logging.error(f"Fatal error while fetching: {repr(e)}")
            break

        if not rows:
            # No more data - advance to end to break
            logging.info("No more rows returned; stopping.")
            break

        all_rows.extend(rows)
        last_ts = rows[-1][0]
        # Advance cursor by one candle to avoid duplicates
        cursor = last_ts + tf_ms

        # Progress info
        bar.set_postfix({"last": datetime.utcfromtimestamp(last_ts/1000).strftime("%Y-%m-%d %H:%M")})
        bar.update(1)

        # Safety sleep (enableRateLimit usually enough; extra guard)
        time.sleep((exchange.rateLimit or 500) / 1000)

        # Stop if the next cursor already beyond the requested end
        if cursor >= to_ms:
            break

    bar.close()
    df = normalize_ohlcv(all_rows)

    # Clip to the exact [since_ms, to_ms)
    if not df.empty:
        df = df[(df["timestamp"] >= pd.to_datetime(since_ms, unit="ms", utc=True)) &
                (df["timestamp"] <  pd.to_datetime(to_ms,    unit="ms", utc=True))]
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    return df

# ----------------------------- CLI ----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect OHLCV data via CCXT and store to data/raw/…"
    )
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange id (ccxt), e.g., binance, bybit, okx")
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair symbol, e.g., BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h", help="CCXT timeframe, e.g., 1m, 5m, 1h, 4h, 1d")
    parser.add_argument("--since", type=str, default=None, help="Start date (UTC) in YYYY-MM-DD, e.g., 2024-01-01")
    parser.add_argument("--to", type=str, default=None, help="End date (UTC) in YYYY-MM-DD (exclusive)")
    parser.add_argument("--limit", type=int, default=1000, help="fetch_ohlcv page size (per request)")
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[2] / "data"),
                        help="Base output folder (default: <repo>/data)")
    parser.add_argument("--fmt", type=str, default="parquet", choices=["parquet","csv"], help="Output format")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")

    args = parser.parse_args()

    exchange_id = args.exchange.lower()
    symbol = args.symbol.upper()
    timeframe = args.timeframe
    base_out = Path(args.out)

    since_ms = parse_date(args.since)
    to_ms = parse_date(args.to)

    logging.info(f"Exchange: {exchange_id} | Symbol: {symbol} | TF: {timeframe} | "
                 f"Since: {args.since or 'auto'} | To: {args.to or 'now'} | fmt: {args.fmt}")

    # init exchange
    exchange = init_exchange(exchange_id)

    # fetch
    df = fetch_ohlcv_incremental(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        since_ms=since_ms,
        to_ms=to_ms,
        limit=args.limit,
        progress=(not args.no_progress),
    )

    if df.empty:
        logging.warning("No data fetched. Nothing to save.")
        return

    # add helpful columns
    df.insert(0, "exchange", exchange_id)
    df.insert(1, "symbol", symbol.replace("/", ""))  # easy to filter later
    df.insert(2, "timeframe", timeframe)

    # save
    outdir = ensure_outdir(base_out, exchange_id, symbol, timeframe)
    save_frame(df, outdir, symbol, timeframe, args.fmt)

    logging.info("Done.")

if __name__ == "__main__":
    main()
