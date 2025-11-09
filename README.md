# Laptop Price Prediction – Experiment (single files)

Tento adresár obsahuje jednotlivé súbory na replikáciu experimentu bez ZIP archívu.

## Reprodukcia
1. Aktivuj virtuálne prostredie a nainštaluj závislosti:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
2. Vlož Kaggle CSV do `data/raw/`.
3. Spusť `notebooks/01_eda.ipynb` → vytvorí `data/processed/clean.csv`.
4. Spusť `notebooks/02_training.ipynb` → vytvorí `outputs/metrics_summary.csv`.
5. Spusť `notebooks/03_explainability.ipynb` → uloží `reports/figures/shap_summary.png`.

## Dáta
- Repozitár obsahuje tri ukážkové datasety v `src/raw/` (`laptop.csv`, `Laptop_price.csv`, `laptop_price - dataset.csv`). Notebook `01_eda.ipynb` ich pri prvom spustení skopíruje/zlúči do `data/raw/` resp. priamo do `data/processed/clean.csv`.
- Ceny z európskeho datasetu sa pri spracovaní konvertujú na INR (1 EUR ≈ 90 INR). Výsledný súbor `data/processed/clean.csv` má 3 192 riadkov so stĺpcami ako `brand`, `cpu`, `ram_gb`, `storage_gb`, `screen_inches`, `rating`, `warranty_years`, `year` a engineered črtami (`storage_type`, `cpu_vendor`, `cpu_gen`, `storage_is_ssd`).

## Automatizované skripty
- Analýza datasetu: `python -m src.dataset_analysis` (generuje štatistiky, grafy a `reports/dataset_analysis_summary.txt`).
- Experiment a porovnanie modelov: `python -m src.experiment_runner` (spustí validácie, uloží `outputs/metrics_*.csv` vrátane `metrics_ci.csv` a `stat_tests.csv`, pripraví grafy a text `reports/experiment_summary.txt`).
- Explainability (SHAP): `python -m src.explainability` (vytvorí `reports/explainability/shap_summary.png`, `shap_feature_importance.csv`, `explainability_summary.txt`).
