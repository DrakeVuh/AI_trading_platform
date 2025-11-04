from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import dataset_dir, parquet_path, normalize_symbol_for_path


def test_dataset_dir_funding(tmp_path):
    out_dir = dataset_dir(tmp_path, "funding", "binance", "BTC/USDT", timeframe=None)

    expected = tmp_path / "funding" / "binance" / "BTCUSDT"
    assert out_dir == expected
    assert out_dir.exists()

    parquet = parquet_path(out_dir)
    assert parquet.name == "data.parquet"
    assert parquet.parent == expected


def test_dataset_dir_ohlcv_month_partition(tmp_path):
    out_dir = dataset_dir(tmp_path, "ohlcv", "binance", "ethusdt", timeframe="1h")

    expected = tmp_path / "ohlcv" / "binance" / "ETHUSDT" / "1h"
    assert out_dir == expected
    assert out_dir.exists()

    month = datetime(2024, 3, 15)
    parquet = parquet_path(out_dir, month)
    assert parquet.name == "2024-03.parquet"
    assert parquet.parent == expected


def test_normalize_symbol_for_path_variants():
    cases = {
        "BTC/USDT": "BTCUSDT",
        "BTC/USDT:USDT": "BTCUSDTUSDT",
        "ETH/USDC.P": "ETHUSDC",
        "sol-perp": "SOLPERP",
    }

    for raw, normalized in cases.items():
        assert normalize_symbol_for_path(raw) == normalized


def test_monthly_partition_across_year_boundary(tmp_path):
    out_dir = dataset_dir(tmp_path, "funding", "binance", "btc/usdt", timeframe=None)

    dec_path = parquet_path(out_dir, datetime(2023, 12, 1))
    jan_path = parquet_path(out_dir, datetime(2024, 1, 1))

    dec_path.write_text("dec")
    jan_path.write_text("jan")

    assert dec_path.name == "2023-12.parquet"
    assert jan_path.name == "2024-01.parquet"
    assert dec_path.exists()
    assert jan_path.exists()


def test_ohlcv_parquet_without_partition(tmp_path):
    out_dir = dataset_dir(tmp_path, "ohlcv", "binance", "ada/usdt", timeframe="1h")
    parquet = parquet_path(out_dir)

    assert parquet.name == "data.parquet"
    assert parquet.parent == out_dir

