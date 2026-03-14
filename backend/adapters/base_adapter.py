from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd


class AdapterError(Exception):
    """Base exception for adapter-related errors."""
    pass


class MissingFileError(AdapterError):
    """Raised when a required file is missing."""
    pass


class InvalidFormatError(AdapterError):
    """Raised when a file exists but its content/columns are invalid."""
    pass


class BaseAdapter(ABC):
    required_files: Tuple[str, ...] = ()

    def __init__(self, files: Dict[str, Union[str, Path]]) -> None:
        self.files: Dict[str, Path] = {name: Path(path) for name, path in files.items()}
        self.raw_data: Dict[str, pd.DataFrame] = {}

    def run(self) -> Dict[str, pd.DataFrame]:
        self.validate_files()
        self.load_raw_data()
        self.validate_content()
        return self.normalize()

    def validate_files(self) -> None:
        for filename in self.required_files:
            if filename not in self.files:
                raise MissingFileError(f"Missing required file: {filename}")

            if not self.files[filename].exists():
                raise MissingFileError(
                    f"File does not exist: {filename} -> {self.files[filename]}"
                )

    def read_csv(self, filename: str, **kwargs: Any) -> pd.DataFrame:
        if filename not in self.files:
            raise MissingFileError(f"Cannot read missing file: {filename}")

        try:
            df = pd.read_csv(self.files[filename], **kwargs)
        except Exception as e:
            raise InvalidFormatError(f"Failed to read CSV file '{filename}': {e}") from e

        if df.empty:
            raise InvalidFormatError(f"CSV file '{filename}' is empty.")

        return df

    @staticmethod
    def require_columns(
        df: pd.DataFrame,
        required_columns: list[str],
        file_label: str,
    ) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise InvalidFormatError(
                f"File '{file_label}' is missing required columns: {missing}"
            )

    @abstractmethod
    def load_raw_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_content(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def normalize(self) -> dict[str, pd.DataFrame]:
        raise NotImplementedError