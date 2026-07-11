import pandas as pd


def add_s2_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe historical features for sensor s2.

    The features are calculated separately for each engine and use
    only the current cycle and past observations.

    Added features
    --------------
    s2_diff_1:
        Difference between the current and previous s2 measurement.

    s2_rolling_mean_5:
        Mean of the current and previous four s2 measurements.

    s2_ema_5:
        Exponentially weighted moving average with span 5.

    s2_expanding_mean:
        Mean of all s2 measurements observed up to the current cycle.

    Parameters
    ----------
    df:
        DataFrame containing at least the columns 'id', 'cycle',
        and 's2'.

    Returns
    -------
    pd.DataFrame
        A sorted copy of the input DataFrame with the new features.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    required_columns = {"id", "cycle", "s2"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.sort_values(["id", "cycle"]).copy()

    grouped_s2 = result.groupby("id")["s2"]

    result["s2_diff_1"] = (
        grouped_s2.diff().fillna(0)
    )

    result["s2_rolling_mean_5"] = (
        grouped_s2
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    result["s2_ema_5"] = (
        grouped_s2
        .transform(
            lambda sensor: sensor.ewm(
                span=5,
                adjust=False
            ).mean()
        )
    )

    result["s2_expanding_mean"] = (
        grouped_s2
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return result