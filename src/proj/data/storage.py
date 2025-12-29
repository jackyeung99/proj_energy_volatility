from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import json

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

    # ---------- existence ----------
    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    # ---------- parquet ----------
    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=True)
        tmp.replace(path)

    # ---------- json ----------
    def read_json(self, key: str):
        """
        Read JSON from local storage.
        Returns dict / list / primitive depending on file contents.
        """
        path = self._path(key)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, obj, key: str) -> None:
        """
        Write JSON atomically (best-effort).
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")

        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)

        tmp.replace(path)


@dataclass
class S3Storage:
    bucket: str
    prefix: str = ""

    def _uri(self, key: str) -> str:
        pref = self.prefix.strip("/")
        if pref:
            return f"s3://{self.bucket}/{pref}/{key}"
        return f"s3://{self.bucket}/{key}"

    def exists(self, key: str) -> bool:
        import s3fs
        fs = s3fs.S3FileSystem()
        return fs.exists(self._uri(key))

    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._uri(key))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        df.to_parquet(self._uri(key), index=False)


def make_storage(cfg: dict) -> Storage:
    s = cfg["storage"]
    backend = s["backend"].lower()

    if backend == "local":
        return LocalStorage(base_dir=Path(s["base_dir"]))

    if backend == "s3":
        return S3Storage(base_uri=s["base_uri"])

    raise ValueError(f"Unknown storage.backend={backend!r}")