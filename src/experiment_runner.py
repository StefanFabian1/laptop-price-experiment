from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.cleaning import build_combined_dataset, collect_clean_datasets
from src.data_prep import build_preprocessor
from src.evaluation import metrics_df
from src.features import add_engineered_features
from src.modeling import get_models

OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("reports/figures")
SUMMARY_PATH = Path("reports/experiment_summary.txt")


def _prepare_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    datasets = collect_clean_datasets()
    if not datasets:
        raise ValueError("Neboli nájdené žiadne datasety na experiment.")

    df_clean = build_combined_dataset()
    source_counts = {name: len(subset) for name, subset in datasets.items()}

    for col in ["rating", "warranty_years", "year", "cpu_physical_cores", "cpu_threads"]:
        median = df_clean[col].median()
        if np.isnan(median):
            median = 0.0
        df_clean[col] = df_clean[col].fillna(median)

    df_features = add_engineered_features(df_clean)

    # Imputuj chýbajúce CPU generácie mediánom (nutné pre scaler)
    if df_features["cpu_gen"].isna().any():
        median_gen = df_features["cpu_gen"].median()
        if np.isnan(median_gen):
            median_gen = 0.0
        df_features["cpu_gen"] = df_features["cpu_gen"].fillna(median_gen)

    df_features["log_price"] = np.log(df_features["price"].clip(lower=1))
    X = df_features.drop(columns=["price", "log_price"])
    y = df_features["log_price"]
    return X, y, df_features, source_counts


def _run_kfold(X: pd.DataFrame, y: pd.Series, models: dict, preprocessor, seed: int = 42) -> pd.DataFrame:
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for model_name, model in models.items():
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            pipeline = Pipeline([("pre", preprocessor), ("model", model)])
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            y_true = np.exp(y.iloc[test_idx])
            y_pred = np.exp(pred)
            metrics = metrics_df(y_true, y_pred, model_name)
            metrics["protocol"] = "kfold"
            metrics["fold"] = fold_idx
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def _time_split_indices(df_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    year_series = df_features["year"]
    cutoff = np.nanquantile(year_series, 0.75)

    # Fallback, ak by kvantil viedol k prázdnemu splitu
    train_mask = year_series <= cutoff
    test_mask = year_series > cutoff
    if test_mask.sum() < 0.1 * len(year_series):
        cutoff = np.nanquantile(year_series, 0.6)
        train_mask = year_series <= cutoff
        test_mask = year_series > cutoff
    if test_mask.sum() == 0:
        raise ValueError("Time-split validácia zlyhala: žiadne dáta v test sete. Skontroluj stĺpec year.")
    return train_mask.values, test_mask.values, float(cutoff)


def _run_time_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_features: pd.DataFrame,
    models: dict,
    preprocessor,
) -> tuple[pd.DataFrame, float, int]:
    train_mask, test_mask, cutoff = _time_split_indices(df_features)
    rows = []
    for model_name, model in models.items():
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])
        pipeline.fit(X.loc[train_mask], y.loc[train_mask])
        pred = pipeline.predict(X.loc[test_mask])
        y_true = np.exp(y.loc[test_mask])
        y_pred = np.exp(pred)
        metrics = metrics_df(y_true, y_pred, model_name)
        metrics["protocol"] = "time_split"
        metrics["fold"] = 0
        metrics["cutoff"] = cutoff
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True), cutoff, int(test_mask.sum())


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["protocol", "model"])
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )
    return grouped


def _save_tables(
    detailed: pd.DataFrame,
    aggregated: pd.DataFrame,
    ci_df: pd.DataFrame | None = None,
    tests_df: pd.DataFrame | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(OUTPUT_DIR / "metrics_detailed.csv", index=False)
    aggregated.to_csv(OUTPUT_DIR / "metrics_mean.csv", index=False)
    if ci_df is not None and not ci_df.empty:
        ci_df.to_csv(OUTPUT_DIR / "metrics_ci.csv", index=False)
    if tests_df is not None and not tests_df.empty:
        tests_df.to_csv(OUTPUT_DIR / "stat_tests.csv", index=False)


def _plot_bar(data: pd.DataFrame, value_col: str, error_col: str | None, title: str, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sorted_data = data.sort_values(value_col).reset_index(drop=True)
    colors = sns.color_palette("crest", n_colors=len(sorted_data))
    x_pos = np.arange(len(sorted_data))
    plt.bar(x_pos, sorted_data[value_col], color=colors)
    if error_col and error_col in data.columns:
        plt.errorbar(
            x_pos,
            sorted_data[value_col],
            yerr=sorted_data[error_col].fillna(0.0),
            fmt="none",
            ecolor="#1b4332",
            capsize=4,
            linewidth=1,
        )
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(value_col)
    plt.xticks(x_pos, sorted_data["model"], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()


def _generate_figures(aggregated: pd.DataFrame) -> dict:
    figures = {}
    kfold = aggregated[aggregated["protocol"] == "kfold"]
    time_split = aggregated[aggregated["protocol"] == "time_split"]

    if not kfold.empty:
        rmse_data = kfold[["model", "RMSE_mean", "RMSE_std"]]
        _plot_bar(rmse_data, "RMSE_mean", "RMSE_std", "K-fold: priemerné RMSE", "rmse_kfold.png")
        figures["rmse_kfold"] = "reports/figures/rmse_kfold.png"

        mape_data = kfold[["model", "MAPE_mean", "MAPE_std"]]
        _plot_bar(mape_data, "MAPE_mean", "MAPE_std", "K-fold: priemerné MAPE", "mape_kfold.png")
        figures["mape_kfold"] = "reports/figures/mape_kfold.png"

        r2_data = kfold[["model", "R2_mean", "R2_std"]]
        _plot_bar(r2_data, "R2_mean", "R2_std", "K-fold: priemerné R\u00b2", "r2_kfold.png")
        figures["r2_kfold"] = "reports/figures/r2_kfold.png"

    if not time_split.empty:
        rmse_ts = time_split[["model", "RMSE_mean"]]
        _plot_bar(rmse_ts, "RMSE_mean", None, "Time-split: RMSE", "rmse_time_split.png")
        figures["rmse_time_split"] = "reports/figures/rmse_time_split.png"

        mape_ts = time_split[["model", "MAPE_mean"]]
        _plot_bar(mape_ts, "MAPE_mean", None, "Time-split: MAPE", "mape_time_split.png")
        figures["mape_time_split"] = "reports/figures/mape_time_split.png"

    return figures


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    clean_values = values[~np.isnan(values)]
    if clean_values.size < 2:
        return np.nan, np.nan
    samples = rng.choice(clean_values, size=(n_resamples, clean_values.size), replace=True).mean(axis=1)
    lower = np.percentile(samples, 100 * alpha / 2.0)
    upper = np.percentile(samples, 100 * (1 - alpha / 2.0))
    return float(lower), float(upper)


def _compute_bootstrap_intervals(kfold_metrics: pd.DataFrame, n_resamples: int = 1000) -> pd.DataFrame:
    rng = default_rng(42)
    rows = []
    for model_name, group in kfold_metrics[kfold_metrics["protocol"] == "kfold"].groupby("model"):
        for metric in ["RMSE", "MAPE", "R2"]:
            vals = group[metric].to_numpy(dtype=float)
            lower, upper = _bootstrap_ci(vals, rng, n_resamples=n_resamples)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_samples": len(vals),
                }
            )
    return pd.DataFrame(rows)


def _paired_tests(kfold_metrics: pd.DataFrame, baseline: str = "ols") -> pd.DataFrame:
    base = (
        kfold_metrics[(kfold_metrics["protocol"] == "kfold") & (kfold_metrics["model"] == baseline)]
        .sort_values("fold")
        .reset_index(drop=True)
    )
    if base.empty:
        return pd.DataFrame()

    rows = []
    baseline_metrics = {
        metric: base[metric].to_numpy(dtype=float)
        for metric in ["RMSE", "MAPE"]
    }

    for model_name, group in (
        kfold_metrics[kfold_metrics["protocol"] == "kfold"]
        .groupby("model")
    ):
        if model_name == baseline:
            continue
        group_sorted = group.sort_values("fold").reset_index(drop=True)
        if len(group_sorted) != len(base):
            continue
        for metric in ["RMSE", "MAPE"]:
            model_vals = group_sorted[metric].to_numpy(dtype=float)
            base_vals = baseline_metrics[metric]
            diffs = model_vals - base_vals
            if np.allclose(diffs.std(ddof=1), 0.0) or len(diffs) < 2:
                t_stat, p_value, cohen_d = np.nan, np.nan, np.nan
            else:
                t_stat, p_value = ttest_rel(model_vals, base_vals)
                cohen_d = diffs.mean() / diffs.std(ddof=1)
            rows.append(
                {
                    "baseline": baseline,
                    "model": model_name,
                    "metric": metric,
                    "mean_difference": diffs.mean(),
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "cohen_d": cohen_d,
                    "n_pairs": len(diffs),
                }
            )
    return pd.DataFrame(rows)


def _build_summary_text(
    source_counts: dict,
    aggregated: pd.DataFrame,
    figures: dict,
    time_split_cutoff: float | None,
    time_split_test_size: int | None,
    ci_df: pd.DataFrame,
    tests_df: pd.DataFrame,
) -> str:
    kfold = aggregated[aggregated["protocol"] == "kfold"].sort_values("RMSE_mean")
    time_split = aggregated[aggregated["protocol"] == "time_split"].sort_values("RMSE_mean")

    best_kfold = kfold.iloc[0] if not kfold.empty else None
    best_time = time_split.iloc[0] if not time_split.empty else None
    baseline = kfold[kfold["model"] == "ols"].iloc[0] if "ols" in kfold["model"].values else None

    def fmt_metrics(row):
        return (
            f"RMSE={row['RMSE_mean']:.0f}±{row['RMSE_std']:.0f}, "
            f"MAPE={row['MAPE_mean']:.2f}%, "
            f"R\u00b2={row['R2_mean']:.3f}"
        )

    text = [
        "Experiment summary",
        "-------------------",
        f"Počet záznamov (spolu): {sum(source_counts.values())}",
        "Zdroje:",
    ]
    text.extend(f"  - {name}: {count}" for name, count in source_counts.items())

    if best_kfold is not None:
        ci_row = ci_df[(ci_df["model"] == best_kfold["model"]) & (ci_df["metric"] == "RMSE")]
        ci_text = ""
        if not ci_row.empty:
            ci_lower = ci_row.iloc[0]["ci_lower"]
            ci_upper = ci_row.iloc[0]["ci_upper"]
            if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                ci_text = f" (95 % CI: {ci_lower:.0f} – {ci_upper:.0f})"
        text.append(
            f"Najlepší model (K-fold): {best_kfold['model']} "
            f"→ {fmt_metrics(best_kfold)}{ci_text}"
        )
    if best_time is not None:
        cutoff_info = ""
        if time_split_cutoff is not None:
            cutoff_info = f"(cutoff year ≤ {int(time_split_cutoff)}) "
        text.append(
            f"Najlepší model (time-split {cutoff_info}): "
            f"{best_time['model']} → RMSE={best_time['RMSE_mean']:.0f}, "
            f"MAPE={best_time['MAPE_mean']:.2f}%, R\u00b2={best_time['R2_mean']:.3f}"
        )
    if baseline is not None and best_kfold is not None:
        improvement = baseline["RMSE_mean"] - best_kfold["RMSE_mean"]
        text.append(
            f"Zlepšenie oproti OLS (RMSE): {improvement:.0f} ("
            f"{improvement / baseline['RMSE_mean'] * 100:.1f} %)."
        )

    if not kfold.empty:
        text.append("\nPoradie modelov (K-fold, podľa RMSE):")
        for row in kfold[["model", "RMSE_mean", "MAPE_mean", "R2_mean"]].itertuples(index=False):
            text.append(
                f"  • {row.model} → RMSE={row.RMSE_mean:.0f}, MAPE={row.MAPE_mean:.2f}%, R\u00b2={row.R2_mean:.3f}"
            )
    if not time_split.empty:
        text.append("\nPoradie modelov (time-split, podľa RMSE):")
        for row in time_split[["model", "RMSE_mean", "MAPE_mean", "R2_mean"]].itertuples(index=False):
            text.append(
                f"  • {row.model} → RMSE={row.RMSE_mean:.0f}, MAPE={row.MAPE_mean:.2f}%, R\u00b2={row.R2_mean:.3f}"
            )

    if not ci_df.empty:
        text.append("\nBootstrap 95 % CI (K-fold):")
        for row in ci_df[ci_df["metric"] == "RMSE"].itertuples(index=False):
            if np.isnan(row.ci_lower) or np.isnan(row.ci_upper):
                continue
            text.append(
                f"  • {row.model} – RMSE CI: {row.ci_lower:.0f} – {row.ci_upper:.0f} (n={row.n_samples})"
            )

    if not tests_df.empty:
        text.append("\nŠtatistické testy (porovnanie s OLS, K-fold):")
        for metric in ["RMSE", "MAPE"]:
            subset = tests_df[tests_df["metric"] == metric]
            if subset.empty:
                continue
            text.append(f"  {metric}:")
            for row in subset.itertuples(index=False):
                significance = "signifikantné" if row.p_value < 0.05 else "nesignifikantné"
                direction = "nižšie" if row.mean_difference < 0 else "vyššie"
                text.append(
                    f"    • {row.model} vs {row.baseline}: Δ={row.mean_difference:.0f} ({direction}), "
                    f"p={row.p_value:.4f}, d={row.cohen_d:.2f} → {significance}"
                )

    text.append("\nGrafy:")
    for name, path in figures.items():
        text.append(f"  - {name}: {path}")

    text.append("\nPoznámky:")
    text.append(
        "- Time-split validácia využíva kvantil roku (0.75); ak bolo testovacích dát málo, cutoff sa posunul na 0.6."
    )
    text.append(
        "- Výsledky sú v INR, log-transformácia ceny sa používala len počas tréningu; metriky sú vypočítané v pôvodnom mierke."
    )
    if time_split_cutoff is not None and time_split_test_size is not None:
        text.append(f"- Time-split testovacia množina obsahovala {time_split_test_size} záznamov (rok > {int(time_split_cutoff)}).")

    return "\n".join(text)


def _write_summary(content: str) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    X, y, df_features, source_counts = _prepare_dataset()
    preprocessor = build_preprocessor()
    models = get_models(random_state=42)

    kfold_metrics = _run_kfold(X, y, models, preprocessor)

    try:
        time_split_metrics, cutoff, test_size = _run_time_split(X, y, df_features, models, preprocessor)
    except ValueError as err:
        time_split_metrics = pd.DataFrame()
        cutoff = None
        test_size = None
        print(f"Time-split validácia preskočená: {err}")

    metrics_combined = pd.concat([kfold_metrics, time_split_metrics], ignore_index=True, sort=False)
    aggregated = _aggregate_metrics(metrics_combined)
    ci_df = _compute_bootstrap_intervals(metrics_combined)
    tests_df = _paired_tests(metrics_combined)

    _save_tables(metrics_combined, aggregated, ci_df=ci_df, tests_df=tests_df)
    figures = _generate_figures(aggregated)
    summary_text = _build_summary_text(
        source_counts,
        aggregated,
        figures,
        cutoff,
        test_size,
        ci_df,
        tests_df,
    )
    _write_summary(summary_text)

    print("Experiment dokončený.")
    print(summary_text)


if __name__ == "__main__":
    main()

