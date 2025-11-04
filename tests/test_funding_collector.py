import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

import pandas as pd
import pytest
import responses

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.collectors import funding_collector


def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@responses.activate
def test_funding_collector_paginates_and_dedupes(tmp_path):
    logger = funding_collector.logger
    handler = logger.handlers[0]
    original_stream = handler.stream
    stream = io.StringIO()
    handler.stream = stream

    try:
        url = funding_collector.BINANCE_FUTURES_REST
        symbol = "BTCUSDT"

        dec_time = datetime(2023, 12, 31, 16, tzinfo=timezone.utc)
        jan_time = dec_time + timedelta(hours=8)
        jan_time2 = jan_time + timedelta(hours=8)

        responses.add(
            responses.GET,
            url,
            json=[
                {
                    "symbol": symbol,
                    "fundingRate": "0.0001",
                    "fundingTime": _ms(dec_time),
                    "nextFundingTime": _ms(jan_time),
                },
                {
                    "symbol": symbol,
                    "fundingRate": "0.0002",
                    "fundingTime": _ms(jan_time),
                    "nextFundingTime": _ms(jan_time2),
                },
            ],
            status=200,
        )

        responses.add(
            responses.GET,
            url,
            json=[
                {
                    "symbol": symbol,
                    "fundingRate": "0.0002",
                    "fundingTime": _ms(jan_time),
                    "nextFundingTime": _ms(jan_time2),
                },
                {
                    "symbol": symbol,
                    "fundingRate": "0.0003",
                    "fundingTime": _ms(jan_time2),
                    "nextFundingTime": _ms(jan_time2 + timedelta(hours=8)),
                },
            ],
            status=200,
        )

        since = datetime(2023, 12, 31, 0, tzinfo=timezone.utc)
        until = datetime(2024, 1, 2, 0, tzinfo=timezone.utc)
        data_root = tmp_path / "data"

        df = funding_collector.collect_funding_rates(
            "BTC/USDT",
            since=since,
            until=until,
            data_root=data_root,
            funding_interval_hours=8,
            bucket="1h",
            use_ccxt=False,
        )

        assert len(responses.calls) >= 2
        assert len(df) == 3
        assert df["timestamp"].is_monotonic_increasing

        symbol_dir = data_root / "raw" / "crypto" / "binance" / "funding" / "BTCUSDT"
        dec_file = symbol_dir / "2023-12.parquet"
        jan_file = symbol_dir / "2024-01.parquet"

        assert dec_file.exists()
        assert jan_file.exists()

        dec_df = pd.read_parquet(dec_file)
        jan_df = pd.read_parquet(jan_file)

        assert dec_df["timestamp"].is_monotonic_increasing
        assert jan_df["timestamp"].is_monotonic_increasing
        assert dec_df["timestamp"].dt.tz is not None
        assert jan_df["timestamp"].dt.tz is not None
        assert pd.Index(dec_df["timestamp"]).is_unique
        assert pd.Index(jan_df["timestamp"]).is_unique

        logs = [json.loads(line) for line in stream.getvalue().splitlines() if line]
        fetch_events = [entry for entry in logs if entry.get("event") == "fetch_batch"]
        save_events = [entry for entry in logs if entry.get("event") == "save_partition"]

        assert fetch_events, "Expected fetch_batch events to be logged"
        assert save_events, "Expected save_partition events to be logged"

        for entry in fetch_events + save_events:
            assert isinstance(entry["ms"], int)
            assert entry["tf"] == "1h"
            assert entry["symbol"] == "BTC/USDT"
        assert symbol_dir.name == "BTCUSDT"
    finally:
        handler.stream = original_stream


@responses.activate
def test_rows_at_boundary_excluded(tmp_path):
    url = funding_collector.BINANCE_FUTURES_REST
    t0 = datetime(2024, 2, 1, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=8)

    responses.add(
        responses.GET,
        url,
        json=[
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": _ms(t0),
            },
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0002",
                "fundingTime": _ms(t1),
            },
        ],
        status=200,
    )

    data_root = tmp_path / "data"
    df = funding_collector.collect_funding_rates(
        "BTC/USDT",
        since=t0,
        until=t1,
        data_root=data_root,
        funding_interval_hours=8,
        bucket="8h",
        use_ccxt=False,
    )

    assert len(df) == 1
    assert df.iloc[0]["timestamp"] == pd.Timestamp(t0)


@responses.activate
def test_next_funding_fallback(tmp_path):
    url = funding_collector.BINANCE_FUTURES_REST
    base = datetime(2024, 3, 10, 0, tzinfo=timezone.utc)

    responses.add(
        responses.GET,
        url,
        json=[
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": _ms(base),
            }
        ],
        status=200,
    )

    df = funding_collector.collect_funding_rates(
        "BTC/USDT",
        since=base,
        until=base + timedelta(hours=12),
        data_root=tmp_path / "data",
        funding_interval_hours=12,
        bucket="12h",
        use_ccxt=False,
    )

    expected_next = base + timedelta(hours=12)
    assert df.iloc[0]["next_funding_ts"] == pd.Timestamp(expected_next)


@responses.activate
def test_idempotent_runs(tmp_path):
    url = funding_collector.BINANCE_FUTURES_REST
    base = datetime(2024, 4, 1, 0, tzinfo=timezone.utc)
    times = [base + timedelta(hours=8 * i) for i in range(3)]

    payload = [
        {
            "symbol": "BTCUSDT",
            "fundingRate": "0.0001",
            "fundingTime": _ms(ts),
        }
        for ts in times
    ]

    responses.add(responses.GET, url, json=payload, status=200)

    data_root = tmp_path / "data"
    since = base
    until = base + timedelta(hours=24)

    df_first = funding_collector.collect_funding_rates(
        "BTC/USDT",
        since=since,
        until=until,
        data_root=data_root,
        funding_interval_hours=8,
        bucket="8h",
        use_ccxt=False,
    )

    responses.add(responses.GET, url, json=payload, status=200)

    df_second = funding_collector.collect_funding_rates(
        "BTC/USDT",
        since=since,
        until=until,
        data_root=data_root,
        funding_interval_hours=8,
        bucket="8h",
        use_ccxt=False,
    )

    parquet_path = data_root / "raw" / "crypto" / "binance" / "funding" / "BTCUSDT" / "2024-04.parquet"
    stored = pd.read_parquet(parquet_path)

    assert len(df_first) == len(df_second) == len(payload)
    assert len(stored) == len(payload)
    assert stored.iloc[0]["timestamp"] == pd.Timestamp(times[0])
    assert stored.iloc[-1]["timestamp"] == pd.Timestamp(times[-1])

