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
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("raw")


def load_turbine_csv(filepath, low_memory=False):
    """
    Load a turbine sensor CSV with metadata header rows.
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

    try:
        data_ext_names = data.copy()
        data_ext_names.columns = ["datetime"] + ext_names
        data_ext_names["datetime"] = pd.to_datetime(data_ext_names["datetime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        data_ext_names = data_ext_names.dropna(subset=["datetime"])
        data_ext_names = data_ext_names.set_index("datetime")

        # Convert all sensor columns to float
        for col in data_ext_names.columns:
            data_ext_names[col] = pd.to_numeric(data_ext_names[col], errors="coerce")
        data = data_ext_names

    except Exception:
        print(f"Fail to using extended names as columns, falling back to sensor IDs.")
        data_sensor_ids = data.copy()
        data_sensor_ids.columns = ["datetime"] + sensor_ids
        
        data_sensor_ids["datetime"] = pd.to_datetime(data_sensor_ids["datetime"], errors="coerce")
        data_sensor_ids = data_sensor_ids.dropna(subset=["datetime"])
        data_sensor_ids = data_sensor_ids.set_index("datetime")

        # Convert all sensor columns to float
        for col in data_sensor_ids.columns:
            data_sensor_ids[col] = pd.to_numeric(data_sensor_ids[col], errors="coerce")
        data = data_sensor_ids


    data = data.sort_index()
    # print(data.head())
    return data, metadata


def available_files():
    """List all CSV files in the raw/ directory."""
    if not RAW_DIR.exists():
        return []
    return sorted([f for f in RAW_DIR.glob("*.csv")])


def get_sensor_summary(data, metadata):
    """Build a per-sensor summary DataFrame."""
    rows = []
    for sid in data.columns:
        meta = metadata.get(sid, {})
        col_data = data[sid].dropna()
        rows.append({
            "Sensor ID":         sid,
            "Description":       meta.get("description", ""),
            "Units":             meta.get("units", ""),
            "Count":             int(col_data.count()),
            "Mean":              round(float(col_data.mean()), 4) if len(col_data) else np.nan,
            "Std":               round(float(col_data.std()), 4) if len(col_data) > 1 else np.nan,
            "Min":               round(float(col_data.min()), 4) if len(col_data) else np.nan,
            "Max":               round(float(col_data.max()), 4) if len(col_data) else np.nan,
        })
    return pd.DataFrame(rows)


def detect_files():
    """
    Auto-detect file types based on column names.
    Returns dict: {label: filepath}
    """
    files = available_files()
    result = {}
    for f in files:
        try:
            raw = pd.read_csv(f, header=None, nrows=1, low_memory=False)
            cols = raw.iloc[0, 1:].tolist()
            name = f.stem

            # CheckBalance/Piston FIRST — has 31MBA10AE005XQ41 which is unique
            has_31mba_ae = any("31MBA10AE005XQ41" in str(c) for c in cols)
            has_31mka_ce = any("31MKA10CE010XQ41" in str(c) for c in cols)

            if has_31mba_ae:
                label = "GT1 Balance/Piston"
            elif has_31mka_ce:
                label = "GT1 Main"
            elif name.lower().startswith("train"):
                label = f"Train ({name})"
            elif name.lower().startswith("test"):
                label = f"Test ({name})"
            else:
                label = name

            # Avoid duplicates by appending filename
            if label in result:
                label = f"{label} ({f.name})"
            result[label] = str(f)
        except Exception:
            result[f.name] = str(f)
    return result
