import re
import numpy as np
import pandas as pd


def _cpu_generation(text: str) -> float:
    if not isinstance(text, str):
        return np.nan
    text = text.strip()
    match = re.search(r"(\d{1,2})(?:st|nd|rd|th)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"Core\s*i\s*(\d)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"Ryzen\s*(\d)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"M(\d)", text)
    if match:
        return 20.0 + float(match.group(1))  # Apple M rodina
    match = re.search(r"(\d{4,5})", text)
    if match:
        return float(match.group(1))
    return np.nan


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_vendor"] = (
        df["cpu"].str.extract(r"^(Intel|AMD|Apple|Qualcomm)", expand=False).fillna("Other")
    )
    df["cpu_gen"] = df["cpu"].apply(_cpu_generation)
    df["gpu_vendor"] = (
        df["gpu"].str.extract(r"^(NVIDIA|AMD|Intel|Apple)", expand=False).fillna("Other")
    )
    df["cpu_gpu_combo"] = df["cpu_vendor"] + "_" + df["gpu_vendor"]
    storage_col = df.get("storage_type")
    if storage_col is not None:
        df["storage_is_ssd"] = storage_col.astype(str).str.lower().eq("ssd").astype(int)
    else:
        df["storage_is_ssd"] = 0
    return df
