import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUM_COLS = [
    "ram_gb",
    "storage_gb",
    "screen_inches",
    "rating",
    "warranty_years",
    "cpu_physical_cores",
    "cpu_threads",
    "cpu_gen",
    "storage_is_ssd",
    "year",
]
CAT_COLS = ["brand", "cpu", "gpu", "os", "storage_type", "cpu_vendor", "gpu_vendor", "cpu_gpu_combo"]

def build_preprocessor():
    num = Pipeline([("scaler", StandardScaler())])
    cat = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
    return pre
