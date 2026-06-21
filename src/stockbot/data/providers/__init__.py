"""Data providers for market data."""

from stockbot.data.providers.base import BaseDataProvider
from stockbot.data.providers.parquet import ParquetDataProvider

__all__ = [
    "BaseDataProvider",
    "ParquetDataProvider",
]
