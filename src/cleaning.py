import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Model",
    "Price",
    "Ram",
    "SSD",
    "Display",
]

FINAL_COLUMNS = [
    "brand",
    "cpu",
    "gpu",
    "os",
    "storage_type",
    "ram_gb",
    "storage_gb",
    "screen_inches",
    "rating",
    "warranty_years",
    "cpu_physical_cores",
    "cpu_threads",
    "year",
    "price",
]

EUR_TO_INR = 90.0


def load_raw_dataset(raw_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Načíta prvý CSV súbor v danom adresári.
    """
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"V adresári {raw_dir} sa nenašiel žiadny CSV súbor.")
    path = os.path.join(raw_dir, files[0])
    df = pd.read_csv(path)
    return df, path


def _parse_price(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def _parse_ram(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def _parse_storage_amount(text: str) -> float:
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    size = float(match.group(1))
    if "tb" in text:
        size *= 1024
    return size


def _parse_storage_type(text: str) -> str:
    if pd.isna(text):
        return "unknown"
    text = str(text).lower()
    if "ssd" in text:
        return "ssd"
    if "hard" in text or "hdd" in text:
        return "hdd"
    return "other"


def clean_laptop_dataframe(
    df_raw: pd.DataFrame, return_report: bool = False
) -> pd.DataFrame:
    """
    Vyčistí surový dataset notebookov a vytvorí konzistentnú tabľku.
    """
    report: dict = {
        "initial_rows": len(df_raw),
        "initial_columns": list(df_raw.columns),
    }

    missing_base = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
    if missing_base:
        raise ValueError(f"Datasetu chýbajú povinné stĺpce: {missing_base}")

    df = df_raw.copy()

    before_drop = len(df)
    df = df.dropna(subset=["Model", "Price"])
    report["dropped_missing_model_or_price"] = before_drop - len(df)
    report["rows_after_model_price_drop"] = len(df)

    df["brand"] = df["Model"].str.extract(r"^([A-Za-z]+)").fillna("Unknown")

    cpu = df["Generation"].fillna("") if "Generation" in df.columns else ""
    if isinstance(cpu, pd.Series):
        cpu = np.where(cpu.astype(str).str.strip() == "", df.get("Core", "").fillna(""), cpu)
    df["cpu"] = pd.Series(cpu).replace("", np.nan).fillna("Unknown CPU")

    df["gpu"] = df.get("Graphics", "Unknown GPU").fillna("Unknown GPU")
    df["os"] = df.get("OS", "Unknown OS").fillna("Unknown OS")
    df["price"] = _parse_price(df["Price"])
    df["ram_gb"] = _parse_ram(df["Ram"])

    df["storage_gb"] = df["SSD"].apply(_parse_storage_amount)
    df["storage_type"] = df["SSD"].apply(_parse_storage_type)

    df["screen_inches"] = pd.to_numeric(
        df["Display"].astype(str).str.replace(",", "").str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )

    df["year"] = pd.to_numeric(df["Model"].str.extract(r"(20\d{2})")[0], errors="coerce")

    df["rating"] = pd.to_numeric(df.get("Rating"), errors="coerce")
    df["cpu_physical_cores"] = pd.to_numeric(
        df.get("Core", "")
        .astype(str)
        .str.extract(r"(\d+)\s*Cores?")[0],
        errors="coerce",
    )
    df["cpu_threads"] = pd.to_numeric(
        df.get("Core", "")
        .astype(str)
        .str.extract(r"(\d+)\s*Threads?")[0],
        errors="coerce",
    )

    df["warranty_years"] = pd.to_numeric(
        df.get("Warranty", "")
        .astype(str)
        .str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    required = ["price", "ram_gb", "storage_gb", "screen_inches"]
    before_required = len(df)
    df = df.dropna(subset=required)
    report["dropped_missing_required_numeric"] = before_required - len(df)
    report["rows_after_required_numeric"] = len(df)

    num_cols = ["rating", "warranty_years", "year", "cpu_physical_cores", "cpu_threads"]
    filled_numeric = {}
    for col in num_cols:
        missing_before = df[col].isna().sum()
        median = df[col].median()
        if np.isnan(median):
            median = 0.0
        df[col] = df[col].fillna(median)
        filled_numeric[col] = missing_before
    report["filled_numeric_values"] = filled_numeric

    cat_cols = ["brand", "cpu", "gpu", "os", "storage_type"]
    filled_categorical = {}
    for col in cat_cols:
        missing_before = df[col].isna().sum()
        df[col] = df[col].fillna("unknown")
        filled_categorical[col] = missing_before
    report["filled_categorical_values"] = filled_categorical

    df_clean = df[
        [
            "brand",
            "cpu",
            "gpu",
            "os",
            "storage_type",
            "ram_gb",
            "storage_gb",
            "screen_inches",
            "rating",
            "warranty_years",
            "cpu_physical_cores",
            "cpu_threads",
            "year",
            "price",
        ]
    ].copy()

    df_clean = df_clean.reset_index(drop=True)
    report["final_rows"] = len(df_clean)
    if return_report:
        return df_clean, report
    return df_clean


def load_and_clean(raw_dir: str) -> Tuple[pd.DataFrame, str]:
    df_raw, path = load_raw_dataset(raw_dir)
    df_clean = clean_laptop_dataframe(df_raw)
    return df_clean, path


def _clean_speed_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["brand"] = df["Brand"].astype(str).str.strip().replace("", "Unknown")
    df["cpu"] = df["Processor_Speed"].apply(
        lambda x: f"Custom {float(x):.2f}GHz" if pd.notna(x) else "Unknown CPU"
    )
    df["gpu"] = "Unknown GPU"
    df["os"] = "unknown"
    df["storage_type"] = "unknown"
    df["ram_gb"] = pd.to_numeric(df["RAM_Size"], errors="coerce")
    df["storage_gb"] = pd.to_numeric(df["Storage_Capacity"], errors="coerce")
    df["screen_inches"] = pd.to_numeric(df["Screen_Size"], errors="coerce")
    df["price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["rating"] = np.nan
    df["warranty_years"] = np.nan
    df["cpu_physical_cores"] = np.nan
    df["cpu_threads"] = np.nan
    df["year"] = np.nan

    df = df.dropna(subset=["price", "ram_gb", "storage_gb", "screen_inches"])

    cleaned = df[FINAL_COLUMNS].copy()
    cleaned["brand"] = cleaned["brand"].replace("", "Unknown")
    cleaned["cpu"] = cleaned["cpu"].replace("", "Unknown CPU")
    return cleaned.reset_index(drop=True)


def _parse_euro_memory(cell: str) -> Tuple[float, str]:
    if pd.isna(cell):
        return np.nan, "unknown"
    text = str(cell).replace("Flash Storage", "SSD").replace("Storage", "")
    parts = re.split(r"\s*\+\s*", text)
    total = 0.0
    types = []
    for part in parts:
        part_lower = part.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)", part_lower)
        if not match:
            continue
        size = float(match.group(1))
        unit = match.group(2)
        if unit == "tb":
            size *= 1024
        total += size
        if "ssd" in part_lower:
            types.append("ssd")
        elif "hdd" in part_lower:
            types.append("hdd")
        elif "hybrid" in part_lower:
            types.append("hybrid")
        else:
            types.append("other")
    if total == 0:
        return np.nan, "unknown"
    if "ssd" in types:
        storage_type = "ssd"
    elif "hdd" in types:
        storage_type = "hdd"
    elif "hybrid" in types:
        storage_type = "hybrid"
    else:
        storage_type = "other"
    return total, storage_type


def _clean_euro_dataset(df_raw: pd.DataFrame, eur_to_inr: float = EUR_TO_INR) -> pd.DataFrame:
    df = df_raw.copy()
    df["brand"] = df["Company"].astype(str).str.strip().replace("", "Unknown")

    def _cpu_format(row: pd.Series) -> str:
        parts = [
            str(row.get("CPU_Company", "")).strip(),
            str(row.get("CPU_Type", "")).strip(),
        ]
        freq = row.get("CPU_Frequency (GHz)")
        if pd.notna(freq):
            parts.append(f"{freq}GHz")
        formatted = " ".join(part for part in parts if part)
        return formatted if formatted else "Unknown CPU"

    df["cpu"] = df.apply(_cpu_format, axis=1)
    df["ram_gb"] = pd.to_numeric(df["RAM (GB)"], errors="coerce")

    storage_info = df["Memory"].apply(_parse_euro_memory)
    df["storage_gb"] = storage_info.apply(lambda x: x[0])
    df["storage_type"] = storage_info.apply(lambda x: x[1])

    df["screen_inches"] = pd.to_numeric(df["Inches"], errors="coerce")
    df["price"] = pd.to_numeric(df["Price (Euro)"], errors="coerce") * eur_to_inr
    df["gpu"] = (
        df["GPU_Company"].fillna("Unknown").astype(str).str.strip()
        + " "
        + df["GPU_Type"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["gpu"] = df["gpu"].replace("", "Unknown GPU")
    df["os"] = df["OpSys"].fillna("unknown")
    df["rating"] = np.nan
    df["warranty_years"] = np.nan
    df["cpu_physical_cores"] = np.nan
    df["cpu_threads"] = np.nan
    df["year"] = np.nan

    df = df.dropna(subset=["price", "ram_gb", "storage_gb", "screen_inches"])
    df["storage_type"] = df["storage_type"].replace("", "unknown")

    cleaned = df[FINAL_COLUMNS].copy()
    cleaned["brand"] = cleaned["brand"].replace("", "Unknown")
    cleaned["cpu"] = cleaned["cpu"].replace("", "Unknown CPU")
    return cleaned.reset_index(drop=True)


def collect_clean_datasets(primary_raw: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}

    if primary_raw is None:
        try:
            primary_raw, _ = load_raw_dataset("data/raw")
        except (FileNotFoundError, ValueError):
            primary_raw = None

    if primary_raw is not None:
        datasets["primary"] = clean_laptop_dataframe(primary_raw)

    speed_path = Path("src/raw/Laptop_price.csv")
    if speed_path.exists():
        speed_df = pd.read_csv(speed_path)
        datasets["speed"] = _clean_speed_dataset(speed_df)

    euro_path = Path("src/raw/laptop_price - dataset.csv")
    if euro_path.exists():
        euro_df = pd.read_csv(euro_path)
        datasets["euro"] = _clean_euro_dataset(euro_df)

    return datasets


def build_combined_dataset(primary_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    datasets = collect_clean_datasets(primary_raw)
    if not datasets:
        raise ValueError("Neboli nájdené žiadne datasety na zlúčenie.")
    combined = pd.concat(datasets.values(), ignore_index=True)
    return combined
