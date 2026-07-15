from typing import List

import pandas as pd
import pytest

from src.config import EMA_SPAN, ROLLING_WINDOW
from src.preprocessing import add_sensor_history_features


def build_s2_feature_columns(
    rolling_window: int,
    ema_span: int,
) -> List[str]:
    return [
        "s2_diff_1",
        f"s2_rolling_mean_{rolling_window}",
        f"s2_ema_{ema_span}",
        "s2_expanding_mean",
    ]


def make_sample_sensor_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
            "s2": [10.0, 12.0, 11.0, 15.0, 14.0, 20.0, 21.0, 24.0],
        }
    )


def test_expected_history_features_are_created() -> None:
    df = make_sample_sensor_data()

    result = add_sensor_history_features(
        df,
        sensors=["s2"],
        rolling_window=5,
        ema_span=5,
    )

    expected_features = build_s2_feature_columns(
        rolling_window=5,
        ema_span=5,
    )

    for column in expected_features:
        assert column in result.columns


def test_history_features_have_no_missing_values() -> None:
    df = make_sample_sensor_data()

    result = add_sensor_history_features(
        df,
        sensors=["s2"],
    )

    feature_columns = build_s2_feature_columns(
        rolling_window=ROLLING_WINDOW,
        ema_span=EMA_SPAN,
    )

    assert result[
        feature_columns
    ].isna().sum().sum() == 0


def test_first_difference_resets_for_each_engine() -> None:
    df = make_sample_sensor_data()

    result = add_sensor_history_features(df, sensors=["s2"])

    first_rows = result.groupby("id").head(1)

    assert (first_rows["s2_diff_1"] == 0).all()


def test_raw_sensor_values_are_preserved() -> None:
    df = make_sample_sensor_data()

    result = add_sensor_history_features(df, sensors=["s2"])

    expected = (
        df.sort_values(["id", "cycle"])
        .reset_index(drop=True)["s2"]
    )

    actual = result.reset_index(drop=True)["s2"]

    pd.testing.assert_series_equal(actual, expected)


def test_feature_engineering_does_not_change_row_count() -> None:
    df = make_sample_sensor_data()

    result = add_sensor_history_features(df, sensors=["s2"])

    assert len(result) == len(df)


def test_future_rows_do_not_change_past_features() -> None:
    """
    Causal leakage test.

    Features calculated at cycle 3 must be identical whether the
    function sees only cycles 1-3 or the complete future history.
    """
    leakage_test_features = build_s2_feature_columns(
        rolling_window=5,
        ema_span=5,
)
    
    full_history = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5, 6],
            "s2": [10.0, 12.0, 11.0, 20.0, 18.0, 25.0],
        }
    )

    truncated_history = full_history[
        full_history["cycle"] <= 3
    ].copy()

    full_result = add_sensor_history_features(
        full_history,
        sensors=["s2"],
        rolling_window=5,
        ema_span=5,
    )

    truncated_result = add_sensor_history_features(
        truncated_history,
        sensors=["s2"],
        rolling_window=5,
        ema_span=5,
    )

    full_past_features = (
        full_result.loc[
            full_result["cycle"] <= 3,
            leakage_test_features,
        ]
        .reset_index(drop=True)
    )

    truncated_features = (
        truncated_result[
            leakage_test_features
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        full_past_features,
        truncated_features,
    )


def test_empty_sensor_list_raises_error() -> None:
    df = make_sample_sensor_data()

    with pytest.raises(
        ValueError,
        match="At least one sensor must be provided",
    ):
        add_sensor_history_features(df, sensors=[])


def test_invalid_window_raises_error() -> None:
    df = make_sample_sensor_data()

    with pytest.raises(
        ValueError,
        match="rolling_window must be at least 1",
    ):
        add_sensor_history_features(
            df,
            sensors=["s2"],
            rolling_window=0,
        )

        
def test_default_feature_names_follow_project_config() -> None:
    df = make_sample_sensor_data()

    result = add_sensor_history_features(
        df,
        sensors=["s2"],
    )

    assert f"s2_rolling_mean_{ROLLING_WINDOW}" in result.columns
    assert f"s2_ema_{EMA_SPAN}" in result.columns