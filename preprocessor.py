"""
Turbine CSV pre-processor
Handles the 5-row metadata header format used by GT1 sensor exports.
Row 0: Point Name (sensor IDs)
Row 1: Description
Row 2: Extended Name
Row 3: Extended Description
Row 4: Units
Row 5+: Data (datetime | sensor values)
"""

import pandas as pd
import numpy as np



def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        orig_non_na = df[col].notna().sum()
        new_non_na = converted.notna().sum()
        if orig_non_na > 0 and (new_non_na / orig_non_na) < 0.5:
            pass # Keep original string/categorical column
        else:
            df[col] = converted
    return df


def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse a timestamp column robustly. Turbine exports appear in more than one
    format (e.g. "06/16/2026 10:21:00 AM" and ISO "2026-06-16 10:21:00"), so we
    try the known explicit formats first and fall back to pandas inference.
    Whichever attempt yields the most valid (non-NaT) timestamps wins.
    """
    known_formats = ["%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"]

    best = None
    best_valid = -1
    for fmt in known_formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        valid = parsed.notna().sum()
        if valid > best_valid:
            best, best_valid = parsed, valid

    # If no explicit format matched most rows, let pandas infer per-value.
    if best_valid < len(series):
        inferred = pd.to_datetime(series, errors="coerce")
        if inferred.notna().sum() > best_valid:
            best = inferred

    return best


def load_apa_csv(filepath, low_memory=True):
    """
    Load a CSV with metadata header rows.
    Accepts a file path (str/Path) or a file-like object (e.g. StringIO).
    Returns:
        data: cleaned DataFrame with datetime index and float columns
        metadata: dict with descriptions, units per sensor
    """
    raw = pd.read_csv(filepath, header=None, low_memory=low_memory)

    # Extract metadata from rows 0-4
    sensor_ids  = raw.iloc[0, 1:].tolist()
    descrips    = raw.iloc[1, 1:].tolist()
    ext_names   = raw.iloc[2, 1:].tolist()
    ext_descr   = raw.iloc[3, 1:].tolist()
    units       = raw.iloc[4, 1:].tolist()

    # Build sensor → metadata mapping
    metadata = {
        sid: {
            "description":       descrips[i] if i < len(descrips) else "",
            "extended_name":     ext_names[i] if i < len(ext_names) else "",
            "extended_desc":     ext_descr[i] if i < len(ext_descr) else "",
            "units":            units[i] if i < len(units) else "",
        }
        for i, sid in enumerate(sensor_ids)
    }

    # Parse data (rows 5 onward)
    data = raw.iloc[5:, :].copy()
    del raw

    try:
        data_ext_names = data.copy()
        data_ext_names.columns = ["datetime"] + ext_names
        data_ext_names["datetime"] = _parse_datetime(data_ext_names["datetime"])
        data_ext_names = data_ext_names.dropna(subset=["datetime"])
        data_ext_names = data_ext_names.set_index("datetime")

        if data_ext_names.empty:
            raise ValueError("No rows survived datetime parsing with extended names.")

        data_ext_names = _coerce_numeric_columns(data_ext_names)
        data = data_ext_names

    except Exception:
        print(f"Fail to using extended names as columns, falling back to sensor IDs.")
        data_sensor_ids = data.copy()
        data_sensor_ids.columns = ["datetime"] + sensor_ids

        data_sensor_ids["datetime"] = _parse_datetime(data_sensor_ids["datetime"])
        data_sensor_ids = data_sensor_ids.dropna(subset=["datetime"])
        data_sensor_ids = data_sensor_ids.set_index("datetime")

        data_sensor_ids = _coerce_numeric_columns(data_sensor_ids)
        data = data_sensor_ids


    data = data.sort_index()
    # print(data.head())
    return data, metadata


