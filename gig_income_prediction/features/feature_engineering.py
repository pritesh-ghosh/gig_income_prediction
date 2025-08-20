import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering that intentionally keeps target-related signals
    to reproduce the near-perfect R² (~0.9999) you saw earlier.
    """
    df = df.copy()

    # --- parse and derive simple time/shift features (safe even if formats vary) ---
    # if parsing fails, result is NaT -> .dt.* handles as NaN
    df["shift_start_hour"] = pd.to_datetime(df.get("shift_start"), errors="coerce").dt.hour
    df["shift_end_hour"]   = pd.to_datetime(df.get("shift_end"),   errors="coerce").dt.hour
    shift_start_dt = pd.to_datetime(df.get("shift_start"), errors="coerce")
    shift_end_dt   = pd.to_datetime(df.get("shift_end"),   errors="coerce")
    df["shift_duration_hours"] = (shift_end_dt - shift_start_dt).dt.total_seconds() / 3600.0

    # day-of-week from Date if present
    if "Date" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["Date"], errors="coerce").dt.dayofweek
        df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype("float64")

    # --- BIG SIGNAL (leaky on purpose): rolling mean of the TARGET per worker ---
    # Columns present in your sheet: 'Worker_id', 'Date', 'net_earnings'
    # This is computed BEFORE the train/test split → reproduces the 0.9999 R².
    if all(col in df.columns for col in ["Worker_id", "net_earnings"]):
        # ensure stable order (by worker then date if Date exists)
        if "Date" in df.columns:
            df = df.sort_values(["Worker_id", "Date"])
        else:
            df = df.sort_values(["Worker_id"]).reset_index(drop=True)

        df["rolling_mean_7d_net"] = (
            df.groupby("Worker_id")["net_earnings"]
              .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        )

    # --- drop raw datetime columns so sklearn doesn't choke on dtypes ---
    datetime_like = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_like.append(col)
    # also drop text datetime-ish columns explicitly (Date, cash_out_date) if they exist
    for c in ["Date", "cash_out_date"]:
        if c in df.columns and c not in datetime_like:
            # if not already recognized as datetime64, we still drop it to avoid dtype issues
            datetime_like.append(c)

    if datetime_like:
        print(f"Dropping datetime columns: {datetime_like}")
        df = df.drop(columns=datetime_like, errors="ignore")

    # return as-is (we are intentionally NOT dropping money component columns)
    return df
