from pathlib import Path
from typing import Union

import pandas as pd


def get_column_names() -> list[str]:
    """
    Return the standard column names used in the NASA C-MAPSS dataset.

    The dataset contains:
    - engine id
    - cycle number
    - 3 operational settings
    - 21 sensor measurements
    """
    return (
        ["id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )


def load_cmapss_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a C-MAPSS train or test data file.

    Parameters
    ----------
    file_path:
        Path to a C-MAPSS text file such as train_FD001.txt
        or test_FD001.txt.

    Returns
    -------
    pd.DataFrame
        Dataset with descriptive column names.

    Raises
    ------
    FileNotFoundError
        If the provided file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"C-MAPSS data file was not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=get_column_names(),
    )


def load_rul_labels(file_path: Union[str, Path]) -> pd.Series:
    """
    Load the true remaining useful life labels for the test engines.

    Parameters
    ----------
    file_path:
        Path to a RUL file such as RUL_FD001.txt.

    Returns
    -------
    pd.Series
        Remaining useful life values, one value per test engine.

    Raises
    ------
    FileNotFoundError
        If the provided file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"RUL label file was not found: {file_path}"
        )

    rul_labels = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    return rul_labels["RUL"]