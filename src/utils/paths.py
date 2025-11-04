"""Filesystem helpers for dataset storage."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

PartitionHint = Union[date, datetime, str]


def normalize_symbol_for_path(symbol: str) -> str:
    """Normalize trading symbols for filesystem-safe paths."""

    primary = symbol.upper().split(".", 1)[0]
    return "".join(ch for ch in primary if ch.isalnum())


def dataset_dir(
    out_root: Union[str, Path],
    dataset: str,
    exchange: str,
    symbol: str,
    timeframe: Optional[str],
) -> Path:
    """Return the directory path for a dataset output, creating it as needed."""

    root = Path(out_root)
    norm_symbol = normalize_symbol_for_path(symbol)

    parts = [root, dataset, exchange, norm_symbol]
    if timeframe:
        parts.append(timeframe)

    target = Path(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def parquet_path(base_dir: Union[str, Path], partition: Optional[PartitionHint] = None) -> Path:
    """Return the Parquet file path for the given dataset directory.

    When ``partition`` is ``None`` the function returns ``<dir>/data.parquet``.
    With a partition provided (``date``/``datetime``/``YYYY-MM`` string) the function
    returns ``<dir>/<YYYY-MM>.parquet``.
    """

    directory = Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if partition is None:
        return directory / "data.parquet"

    if isinstance(partition, str):
        stamp = partition
    elif isinstance(partition, datetime):
        stamp = partition.strftime("%Y-%m")
    elif isinstance(partition, date):
        stamp = partition.strftime("%Y-%m")
    else:  # pragma: no cover - defensive fallback
        raise TypeError("Unsupported partition type")

    return directory / f"{stamp}.parquet"

