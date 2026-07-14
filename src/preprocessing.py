import pandas as pd
from typing import List
from src.config import EMA_SPAN, ROLLING_WINDOW

def add_sensor_history_features(
    df: pd.DataFrame,
    sensors: List[str],
    rolling_window: int = ROLLING_WINDOW,
    ema_span: int = EMA_SPAN,
) -> pd.DataFrame:
    """
    Add leakage-safe historical features for multiple sensors.

    For every selected sensor, the function creates:

    - one-cycle difference
    - rolling mean
    - exponential moving average
    - expanding mean

    All calculations are performed separately for each engine and use
    only the current and previous observations.

    Parameters
    ----------
    df:
        DataFrame containing 'id', 'cycle', and the selected sensors.

    sensors:
        List of sensor column names, such as ['s2', 's3', 's4'].

    rolling_window:
        Number of cycles used for the rolling mean.

    ema_span:
        Span used for the exponential moving average.

    Returns
    -------
    pd.DataFrame
        A sorted copy of the input DataFrame with new historical
        features.

    Raises
    ------
    ValueError
        If required columns are missing or the sensor list is empty.
    """
    if not sensors:
        raise ValueError("At least one sensor must be provided.")

    required_columns = {"id", "cycle"} | set(sensors)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1.")

    if ema_span < 1:
        raise ValueError("ema_span must be at least 1.")

    result = df.sort_values(["id", "cycle"]).copy()

    for sensor in sensors:
        grouped_sensor = result.groupby("id")[sensor]

        result[f"{sensor}_diff_1"] = (
            grouped_sensor.diff().fillna(0)
        )

        result[f"{sensor}_rolling_mean_{rolling_window}"] = (
            grouped_sensor
            .rolling(
                window=rolling_window,
                min_periods=1
            )
            .mean()
            .reset_index(level=0, drop=True)
        )

        result[f"{sensor}_ema_{ema_span}"] = (
            grouped_sensor.transform(
                lambda values: values.ewm(
                    span=ema_span,
                    adjust=False
                ).mean()
            )
        )

        result[f"{sensor}_expanding_mean"] = (
            grouped_sensor
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return result