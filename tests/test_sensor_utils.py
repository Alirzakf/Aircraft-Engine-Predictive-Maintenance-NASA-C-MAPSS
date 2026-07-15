import pandas as pd
import pytest

from src.config import NEAR_CONSTANT_VARIANCE_THRESHOLD
from src.sensor_utils import (
    calculate_sensor_variance,
    find_low_variance_sensors,
)


def make_sensor_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "s1": [5.0, 5.0, 5.0, 5.0],
            "s2": [1.0, 2.0, 3.0, 4.0],
            "s3": [10.0, 10.0, 10.0, 10.0],
        }
    )


def test_sensor_variance_is_sorted() -> None:
    df = make_sensor_data()

    result = calculate_sensor_variance(
        df,
        sensors=["s1", "s2", "s3"],
    )

    assert result.is_monotonic_increasing


def test_constant_sensors_are_detected() -> None:
    df = make_sensor_data()

    result = find_low_variance_sensors(
        df,
        sensors=["s1", "s2", "s3"],
        threshold=1e-10,
    )

    assert set(result) == {"s1", "s3"}


def test_negative_threshold_raises_error() -> None:
    df = make_sensor_data()

    with pytest.raises(
        ValueError,
        match="threshold must not be negative",
    ):
        find_low_variance_sensors(
            df,
            sensors=["s1"],
            threshold=-1,
        )


def test_default_threshold_follows_project_config() -> None:
    df = make_sensor_data()

    result = find_low_variance_sensors(
        df,
        sensors=["s1", "s2", "s3"],
    )

    assert NEAR_CONSTANT_VARIANCE_THRESHOLD == 1e-10
    assert set(result) == {"s1", "s3"}