# Kapitola 4 – Experiment (krok za krokom)

Tento súbor ťa prevedie vykonaním experimentu podľa `experiment_section4_procedure.md` – prakticky, po krokoch.

---

## Krok 0 – Príprava prostredia
- Nainštaluj Python 3.10+.
- V príkazovom riadku spusti (Windows PowerShell / macOS/Linux Terminal):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Krok 1 – Štruktúra projektu
Vytvor adresáre (ak nepoužívaš tieto priložené súbory):
```
data/raw
data/processed
notebooks
src
outputs
reports/figures
```

---

## Krok 2 – Dáta z Kaggle
1. Stiahni dataset *Laptop Price Dataset (Suman Bera)* z Kaggle.
2. Ulož CSV súbory do `data/raw/`. (Ak použiješ priložený `src/raw/laptop.csv`, notebook `01_eda.ipynb` ho automaticky skopíruje do `data/raw/` pri prvom spustení.)
3. Skontroluj, že aspoň jeden `.csv` súbor v `data/raw/` existuje.

---

## Krok 3 – EDA a čistý dataset
1. Otvor `notebooks/01_eda.ipynb` a spusti všetky bunky.
2. Notebook načíta všetky dostupné CSV (`data/raw/` + `src/raw/`) a zlúči ich do jednotného datasetu.
3. Výstupom je `data/processed/clean.csv`.

> Ak chýbajú niektoré stĺpce podľa očakávanej schémy, notebook to vypíše (je to v poriadku – dataset môže mať inú štruktúru; pokračuj ďalej).

---

## Krok 4 – Tréning a hodnotenie
1. Otvor `notebooks/02_training.ipynb` a spusti všetky bunky.
2. Natrénuje sa OLS, Ridge, Lasso, Random Forest, SVR, XGBoost.
3. Vyhodnotí sa 5-fold K-fold a time-split.
4. Vznikne súbor `outputs/metrics_summary.csv`.

---

## Krok 5 – Explainability (SHAP)
1. Otvor `notebooks/03_explainability.ipynb` a spusti všetky bunky (voliteľné, historická verzia obrázka v `reports/figures/shap_summary.png`).
2. Odporúčaný skript: `python -m src.explainability` → uloží výstupy do `reports/explainability/` (`shap_summary.png`, `shap_feature_importance.csv`, `explainability_summary.txt`).

---

## Krok 6 – Automatizovaný experiment (voliteľné)
- Spusti `python -m src.dataset_analysis` (štatistiky + grafy do `reports/`).
- Spusti `python -m src.experiment_runner` (tréning modelov, metriky do `outputs/`, vrátane `metrics_ci.csv` a `stat_tests.csv`, grafy + text do `reports/`).
- Spusti `python -m src.explainability` (SHAP analýza, výstupy v `reports/explainability/`).

---

## Krok 6 – Artefakty do práce
- Prilož `outputs/metrics_summary.csv` a obrázok z `reports/figures/` (plus ďalšie figúry/boxploty, ak doplníš).
- Ak škola umožní, prilož aj vzorku dát (alebo presný odkaz na Kaggle) a kód (`notebooks/`, `src/`, `requirements.txt`).

---

## Tipy – 95 % CI a štatistika
- Do `02_training.ipynb` môžeš doplniť bootstrap CI a párové *t*-testy (napr. `scipy.stats`). Stačí vziať tabuľku chýb na foldoch a bootstrapovať priemer RMSE/MAPE 1 000×.
