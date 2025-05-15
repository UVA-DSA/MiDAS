import pandas as pd

def remove_out_of_order(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Removes trailing rows where the timestamp goes backwards.
    Keeps rows up to the last monotonically non-decreasing timestamp.
    """
    df = df.copy()
    # Ensure time_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], unit='ms')
    # Compute time differences
    dt = df[time_col].diff()
    # Find the first position where time decreases
    backward = dt[dt < pd.Timedelta(0)]
    if not backward.empty:
        cut_idx = backward.index[0]
        df = df.loc[:cut_idx-1].reset_index(drop=True)
    return df