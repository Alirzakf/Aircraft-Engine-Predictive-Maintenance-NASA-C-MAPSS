from typing import List

import pandas as pd


def calculate_sensor_variance(
    df: pd.DataFrame,
    sensors: List[str],
) -> pd.Series:
    """
    Calculate the variance of selected sensor columns.

    Parameters
    ----------
    df:
        DataFrame containing sensor measurements.

    sensors:
        List of sensor column names.

    Returns
    -------
    pd.Series
        Sensor variances sorted from lowest to highest.

    Raises
    ------
    ValueError
        If the sensor list is empty or required columns are missing.
    """
    if not sensors:
        raise ValueError("At least one sensor must be provided.")

    missing_sensors = set(sensors) - set(df.columns)

    if missing_sensors:
        raise ValueError(
            f"Missing sensor columns: {sorted(missing_sensors)}"
        )

    return df[sensors].var().sort_values()


def find_low_variance_sensors(
    df: pd.DataFrame,
    sensors: List[str],
    threshold: float = 1e-10,
) -> List[str]:
    """
    Identify sensors whose variance is below a selected threshold.

    Parameters
    ----------
    df:
        DataFrame containing sensor measurements.

    sensors:
        List of sensor column names.

    threshold:
        Maximum raw variance used to classify a sensor as effectively
        constant. Because variance depends on measurement scale, this
        threshold should be used only to detect near-zero-variance sensors,
        not for general feature selection.

    Returns
    -------
    List[str]
        Sensor names with variance below the threshold.

    Raises
    ------
    ValueError
        If the threshold is negative.
    """
    if threshold < 0:
        raise ValueError("threshold must not be negative.")

    sensor_variance = calculate_sensor_variance(
        df=df,
        sensors=sensors,
    )

    return sensor_variance[
        sensor_variance < threshold
    ].index.tolist()