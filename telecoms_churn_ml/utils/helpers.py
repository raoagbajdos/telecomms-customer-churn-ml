"""Helper utility functions."""

import os
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    # Go up from telecoms_churn_ml/utils/helpers.py to project root
    return current_file.parent.parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(filepath: Union[str, Path]) -> Path:
    """Validate that a file exists."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return filepath


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Get file size in megabytes."""
    filepath = Path(filepath)
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0.0


def clean_column_name(column_name: str) -> str:
    """Clean and standardize column names."""
    return (column_name
            .lower()
            .replace(' ', '_')
            .replace('-', '_')
            .replace('.', '_')
            .strip('_'))


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: Union[int, float], thousands_sep: str = ',') -> str:
    """Format a number with thousands separator."""
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:,.2f}"
