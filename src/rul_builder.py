from typing import Optional

import pandas as pd


def add_train_rul(
    df: pd.DataFrame,
    rul_cap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add the Remaining Useful Life target to C-MAPSS training data.

    Linear RUL is calculated for engine i at cycle t as:

        RUL = final observed cycle - current cycle

    When ``rul_cap`` is provided, the target is capped at that value:

        capped RUL = min(linear RUL, rul_cap)

    The final observed cycle is used only to construct the training
    target. It is not retained as a feature in the returned DataFrame.

    Parameters
    ----------
    df:
        Training DataFrame containing at least the columns
        ``id`` and ``cycle``.

    rul_cap:
        Optional maximum RUL value. When ``None``, the original linear
        RUL target is returned. When a positive integer is provided,
        RUL values above that threshold are capped.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an added ``RUL`` column.

    Raises
    ------
    ValueError
        If required columns are missing or ``rul_cap`` is not positive.
    """
    required_columns = {"id", "cycle"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if rul_cap is not None and rul_cap <= 0:
        raise ValueError(
            "rul_cap must be a positive integer or None."
        )

    result = df.copy()

    final_cycle = (
        result.groupby("id")["cycle"]
        .transform("max")
    )

    result["RUL"] = final_cycle - result["cycle"]

    if rul_cap is not None:
        result["RUL"] = result["RUL"].clip(
            upper=rul_cap
        )

    return result