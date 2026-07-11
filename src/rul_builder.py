import pandas as pd


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the Remaining Useful Life target to C-MAPSS training data.

    For each engine, RUL is calculated as:

        RUL = final observed cycle - current cycle

    The final observed cycle is used only to construct the training target.
    It must not be included as a model feature.

    Parameters
    ----------
    df:
        Training DataFrame containing at least the columns
        'id' and 'cycle'.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an added 'RUL' column.

    Raises
    ------
    ValueError
        If the required columns are missing.
    """
    required_columns = {"id", "cycle"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy()

    final_cycle = result.groupby("id")["cycle"].transform("max")
    result["RUL"] = final_cycle - result["cycle"]

    return result