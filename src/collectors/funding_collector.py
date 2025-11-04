"""Collector for Binance funding rates."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ccxt
import pandas as pd

from src.utils.http import request_json
from src.utils.logger import get_logger, log_event
from src.utils.paths import normalize_symbol_for_path, parquet_path


logger = get_logger(__name__)

BINANCE_FUTURES_REST = "https://fapi.binance.com/fapi/v1/fundingRate"
DEFAULT_EXCHANGE = "binance"
DATASET = "funding"
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2] / "data"
PAGE_LIMIT = 1000


def _to_utc_ms(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return int(timestamp.timestamp() * 1000)


def _ensure_chronology(since: datetime, until: datetime) -> Tuple[datetime, datetime]:
    if since >= until:
        raise ValueError("'since' must be earlier than 'until'.")
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return since, until


def _init_ccxt_exchange() -> Optional[ccxt.Exchange]:
    try:
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        if not getattr(exchange, "has", {}).get("fetchFundingRateHistory", False):
            return None
        return exchange
    except Exception:
        return None


def _extract_record(entry: Dict[str, any], symbol: str) -> Dict[str, any]:
    info = entry.get("info", entry)
    funding_ms = int(info.get("fundingTime")) if "fundingTime" in info else int(entry.get("timestamp"))
    next_ms = None
    if "nextFundingTime" in info and info["nextFundingTime"]:
        next_ms = int(info["nextFundingTime"])
    elif "nextFundingTime" in entry and entry["nextFundingTime"]:
        next_ms = int(entry["nextFundingTime"])

    rate_value = entry.get("fundingRate") or info.get("fundingRate")
    rate = float(rate_value) if rate_value is not None else None

    return {
        "timestamp": funding_ms,
        "symbol": symbol,
        "funding_rate": rate,
        "next_funding_ts": next_ms,
    }


def _fetch_via_ccxt(
    symbol: str,
    since_ms: int,
    until_ms: int,
    bucket: str,
    limit: int = PAGE_LIMIT,
    exchange_label: str = DEFAULT_EXCHANGE,
) -> List[Dict[str, any]]:
    ccxt_exchange = _init_ccxt_exchange()
    if ccxt_exchange is None:
        return []

    records: List[Dict[str, any]] = []
    cursor = since_ms

    while cursor < until_ms:
        params = {"endTime": until_ms}
        start_time = time.perf_counter()
        try:
            batch = ccxt_exchange.fetchFundingRateHistory(symbol, since=cursor, limit=limit, params=params)  # type: ignore[attr-defined]
        except Exception:
            return []
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        rows = batch or []
        log_event(
            logger,
            service="collector",
            event="fetch_batch",
            dataset=DATASET,
            exchange=exchange_label,
            symbol=symbol,
            tf=bucket,
            rows=len(rows),
            ms=elapsed_ms,
        )

        if not rows:
            break

        appended = False
        for entry in rows:
            record = _extract_record(entry, symbol)
            if record["timestamp"] >= until_ms:
                continue
            records.append(record)
            appended = True

        last_ts_in_batch = max(int(_extract_record(entry, symbol)["timestamp"]) for entry in rows)
        if appended:
            last_ts = records[-1]["timestamp"]
        else:
            last_ts = last_ts_in_batch

        if last_ts < cursor:
            break
        cursor = last_ts + 1

    return records


def _fetch_via_http(
    symbol: str,
    since_ms: int,
    until_ms: int,
    bucket: str,
    limit: int = PAGE_LIMIT,
    exchange_label: str = DEFAULT_EXCHANGE,
) -> List[Dict[str, any]]:
    records: List[Dict[str, any]] = []
    cursor = since_ms
    http_symbol = symbol.replace("/", "").upper()

    while cursor < until_ms:
        params = {
            "symbol": http_symbol,
            "limit": limit,
            "startTime": cursor,
            "endTime": until_ms - 1,
        }

        batch, elapsed_ms = request_json(BINANCE_FUTURES_REST, params=params)
        rows = batch or []

        log_event(
            logger,
            service="collector",
            event="fetch_batch",
            dataset=DATASET,
            exchange=exchange_label,
            symbol=symbol,
            tf=bucket,
            rows=len(rows),
            ms=elapsed_ms,
        )

        if not rows:
            break

        appended = False
        for info in rows:
            funding_ms = int(info["fundingTime"])
            if funding_ms >= until_ms:
                continue
            next_ms = int(info["nextFundingTime"]) if info.get("nextFundingTime") else None
            rate = float(info.get("fundingRate")) if info.get("fundingRate") is not None else None
            records.append(
                {
                    "timestamp": funding_ms,
                    "symbol": symbol,
                    "funding_rate": rate,
                    "next_funding_ts": next_ms,
                }
            )
            appended = True

        last_ts_in_batch = max(int(info["fundingTime"]) for info in rows)
        if appended:
            last_ts = records[-1]["timestamp"]
        else:
            last_ts = last_ts_in_batch

        if last_ts < cursor:
            break
        cursor = last_ts + 1

    return records


def _prepare_dataframe(entries: Iterable[Dict[str, any]], *, funding_interval_hours: int, bucket: str, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(list(entries))
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")

    if "next_funding_ts" in df:
        df["next_funding_ts"] = pd.to_datetime(df["next_funding_ts"], unit="ms", utc=True, errors="coerce")
        missing = df["next_funding_ts"].isna()
        if missing.any():
            fallback_delta = timedelta(hours=funding_interval_hours)
            df.loc[missing, "next_funding_ts"] = df.loc[missing, "timestamp"] + fallback_delta
    else:
        fallback_delta = timedelta(hours=funding_interval_hours)
        df["next_funding_ts"] = df["timestamp"] + fallback_delta

    if "funding_rate" in df:
        extreme = df["funding_rate"].abs() > 0.05
        for ts, rate in df.loc[extreme, ["timestamp", "funding_rate"]].itertuples(index=False):
            logger.warning(
                "Funding rate spike | symbol=%s ts=%s rate=%s",
                symbol,
                ts.isoformat(),
                rate,
            )

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "symbol", "funding_rate", "next_funding_ts"]]


def _symbol_directory(data_root: Path, symbol: str, exchange: str) -> Path:
    normalized = normalize_symbol_for_path(symbol)
    target = data_root / "raw" / "crypto" / exchange / DATASET / normalized
    target.mkdir(parents=True, exist_ok=True)
    return target


def _output_path(symbol_dir: Path, partition_key: Optional[str], output_format: str) -> Path:
    if output_format == "parquet":
        return parquet_path(symbol_dir, partition_key)

    symbol_dir.mkdir(parents=True, exist_ok=True)
    if partition_key is None:
        return symbol_dir / "data.csv"

    return symbol_dir / f"{partition_key}.csv"


def _read_existing(path: Path, output_format: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    if output_format == "parquet":
        return pd.read_parquet(path)

    df = pd.read_csv(path, parse_dates=["timestamp", "next_funding_ts"])
    for column in ("timestamp", "next_funding_ts"):
        if column in df:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    return df


def _write_dataframe(path: Path, df: pd.DataFrame, output_format: str) -> None:
    if output_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _save_partitions(
    symbol_dir: Path,
    df: pd.DataFrame,
    symbol: str,
    bucket: str,
    exchange: str,
    output_format: str,
    partition: str,
) -> None:
    if df.empty:
        return

    df = df.copy()
    if partition == "monthly":
        df["partition_key"] = df["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
        grouped = df.groupby("partition_key")
    else:
        df["partition_key"] = None
        grouped = [(None, df.drop(columns="partition_key"))]

    for partition_key, group in grouped:
        if partition_key is None:
            data = group.copy()
        else:
            data = group.drop(columns="partition_key")

        path = _output_path(symbol_dir, partition_key, output_format)

        start = time.perf_counter()
        existing = _read_existing(path, output_format)
        if existing is not None:
            combined = pd.concat([existing, data], ignore_index=True)
        else:
            combined = data

        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        _write_dataframe(path, combined, output_format)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        log_event(
            logger,
            service="collector",
            event="save_partition",
            dataset=DATASET,
            exchange=exchange,
            symbol=symbol,
            tf=bucket,
            rows=len(combined),
            ms=elapsed_ms,
        )


def collect_funding_rates(
    symbol: str,
    since: datetime,
    until: datetime,
    data_root: Path = DEFAULT_BASE_DIR,
    funding_interval_hours: int = 8,
    bucket: Optional[str] = None,
    use_ccxt: bool = True,
    exchange: str = DEFAULT_EXCHANGE,
    output_format: str = "parquet",
    partition: str = "monthly",
) -> pd.DataFrame:
    symbol = symbol.upper()
    since, until = _ensure_chronology(since, until)

    if bucket is None:
        bucket = f"{funding_interval_hours}h"

    output_format = output_format.lower()
    partition = partition.lower()

    if output_format not in {"parquet", "csv"}:
        raise ValueError("output_format must be either 'parquet' or 'csv'")
    if partition not in {"monthly", "none"}:
        raise ValueError("partition must be either 'monthly' or 'none'")

    since_ms = _to_utc_ms(since)
    until_ms = _to_utc_ms(until)

    entries: List[Dict[str, any]] = []
    if use_ccxt:
        entries = _fetch_via_ccxt(symbol, since_ms, until_ms, bucket, exchange_label=exchange)

    if not entries:
        entries = _fetch_via_http(symbol, since_ms, until_ms, bucket, exchange_label=exchange)

    df = _prepare_dataframe(entries, funding_interval_hours=funding_interval_hours, bucket=bucket, symbol=symbol)

    if not df.empty:
        cutoff = until.astimezone(timezone.utc)
        df = df[df["timestamp"] < cutoff]

    symbol_dir = _symbol_directory(data_root, symbol, exchange)
    _save_partitions(symbol_dir, df, symbol, bucket, exchange, output_format, partition)

    return df


def _parse_datetime(value: str) -> datetime:
    try:
        dt = pd.to_datetime(value, utc=True)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid datetime format: {value}") from exc

    if dt.tzinfo is None:
        dt = dt.tz_localize(timezone.utc)

    return dt.to_pydatetime()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Binance funding rates")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g., BTC/USDT")
    parser.add_argument("--since", required=True, type=_parse_datetime, help="Start datetime (ISO, UTC)")
    parser.add_argument("--until", "--to", required=True, dest="until", type=_parse_datetime, help="End datetime (ISO, UTC, exclusive)")
    parser.add_argument("--data-root", "--out", default=str(DEFAULT_BASE_DIR), dest="data_root", help="Base data directory")
    parser.add_argument(
        "--funding-interval-hours",
        type=int,
        default=8,
        help="Fallback funding interval in hours when nextFundingTime is absent",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Logical timeframe label for logging (default derived from interval hours)",
    )
    parser.add_argument(
        "--disable-ccxt",
        action="store_true",
        help="Disable ccxt funding history and use REST fallback only",
    )
    parser.add_argument(
        "--exchange",
        default=DEFAULT_EXCHANGE,
        help="Exchange label for logging (default: binance)",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format",
    )
    parser.add_argument(
        "--partition",
        choices=("none", "monthly"),
        default="monthly",
        help="Partitioning strategy",
    )

    args = parser.parse_args()

    symbol = args.symbol
    since: datetime = args.since
    until: datetime = args.until
    data_root = Path(args.data_root)
    interval_hours = args.funding_interval_hours
    bucket = args.bucket
    use_ccxt = not args.disable_ccxt
    exchange = args.exchange
    output_format = args.format
    partition = args.partition

    logger.info(
        "Starting funding collection",
        extra={
            "service": "collector",
            "event": "start",
            "dataset": DATASET,
        "exchange": exchange,
            "symbol": symbol,
            "tf": bucket or f"{interval_hours}h",
            "rows": 0,
            "ms": 0,
        },
    )

    df = collect_funding_rates(
        symbol,
        since,
        until,
        data_root=data_root,
        funding_interval_hours=interval_hours,
        bucket=bucket,
        use_ccxt=use_ccxt,
        exchange=exchange,
        output_format=output_format,
        partition=partition,
    )

    logger.info(
        "Completed funding collection",
        extra={
            "service": "collector",
            "event": "complete",
            "dataset": DATASET,
        "exchange": exchange,
            "symbol": symbol,
            "tf": bucket or f"{interval_hours}h",
            "rows": len(df),
            "ms": 0,
        },
    )


__all__ = ["collect_funding_rates", "main"]


if __name__ == "__main__":
    main()

