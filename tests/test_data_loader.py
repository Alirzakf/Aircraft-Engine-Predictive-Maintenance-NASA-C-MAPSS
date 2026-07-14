from pathlib import Path

import pytest

from src.data_loader import (
    get_column_names,
    load_cmapss_data,
    load_rul_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def test_column_names_are_complete() -> None:
    columns = get_column_names()

    assert len(columns) == 26
    assert columns[:5] == [
        "id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ]
    assert columns[-1] == "s21"


def test_load_fd001_training_data() -> None:
    train_path = RAW_DATA_DIR / "train_FD001.txt"

    df = load_cmapss_data(train_path)

    assert df.shape == (20631, 26)
    assert df["id"].nunique() == 100
    assert df.columns.tolist() == get_column_names()
    assert df.isna().sum().sum() == 0


def test_load_fd001_rul_labels() -> None:
    rul_path = RAW_DATA_DIR / "RUL_FD001.txt"

    rul = load_rul_labels(rul_path)

    assert len(rul) == 100
    assert rul.name == "RUL"
    assert rul.isna().sum() == 0


def test_missing_data_file_raises_error() -> None:
    missing_path = RAW_DATA_DIR / "missing_file.txt"

    with pytest.raises(FileNotFoundError):
        load_cmapss_data(missing_path)