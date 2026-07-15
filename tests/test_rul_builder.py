import pandas as pd
import pytest

from src.rul_builder import add_train_rul


def make_sample_engine_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "s2": [10.0, 11.0, 12.0, 20.0, 21.0],
        }
    )


def test_train_rul_is_calculated_per_engine() -> None:
    df = make_sample_engine_data()

    result = add_train_rul(df)

    assert result["RUL"].tolist() == [2, 1, 0, 1, 0]


def test_final_cycle_has_zero_rul() -> None:
    df = make_sample_engine_data()

    result = add_train_rul(df)

    final_rows = result.groupby("id").tail(1)

    assert (final_rows["RUL"] == 0).all()


def test_future_lifetime_column_is_not_retained() -> None:
    df = make_sample_engine_data()

    result = add_train_rul(df)

    assert "max_cycle" not in result.columns
    assert "RUL" in result.columns


def test_input_dataframe_is_not_modified() -> None:
    df = make_sample_engine_data()
    original = df.copy(deep=True)

    add_train_rul(df)

    pd.testing.assert_frame_equal(df, original)


def test_missing_required_column_raises_error() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "s2": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        add_train_rul(df)
        

def test_capped_rul_limits_large_values() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 200],
        }
    )

    result = add_train_rul(
        df,
        rul_cap=125,
    )

    assert result["RUL"].tolist() == [
        125,
        125,
        125,
        0,
    ]


def test_capped_rul_preserves_values_below_cap() -> None:
    df = make_sample_engine_data()

    linear_result = add_train_rul(df)

    capped_result = add_train_rul(
        df,
        rul_cap=125,
    )

    pd.testing.assert_series_equal(
        linear_result["RUL"],
        capped_result["RUL"],
    )


def test_capped_rul_never_exceeds_cap() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "cycle": [1, 50, 200],
        }
    )

    result = add_train_rul(
        df,
        rul_cap=125,
    )

    assert result["RUL"].max() == 125
    assert (result["RUL"] <= 125).all()


@pytest.mark.parametrize(
    "invalid_cap",
    [0, -1, -125],
)
def test_invalid_rul_cap_raises_error(
    invalid_cap: int,
) -> None:
    df = make_sample_engine_data()

    with pytest.raises(
        ValueError,
        match="rul_cap must be a positive integer or None",
    ):
        add_train_rul(
            df,
            rul_cap=invalid_cap,
        )