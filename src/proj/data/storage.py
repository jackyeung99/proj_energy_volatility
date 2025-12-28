from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

class Storage:
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def read_parquet(self, key: str) -> pd.DataFrame:
        raise NotImplementedError

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalStorage(Storage):
    base_dir: Path

    def _path(self, key: str) -> Path:
        return (self.base_dir / key).resolve()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)


@dataclass(frozen=True)
class URIStorage(Storage):
    """
    Works for s3://..., gcs://..., etc. if the right fsspec backend is installed
    (e.g. s3fs for S3).
    """
    base_uri: str  # e.g. "s3://my-bucket/proj-data"

    def _uri(self, key: str) -> str:
        return f"{self.base_uri.rstrip('/')}/{key.lstrip('/')}"

    def exists(self, key: str) -> bool:
        pass

    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._uri(key))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        df.to_parquet(self._uri(key), index=False)
