from pathlib import Path
from textwrap import dedent
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cleaning import collect_clean_datasets
from src.features import add_engineered_features

FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/dataset_analysis_summary.txt")


def _describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    percent = missing / len(df) * 100
    summary = pd.DataFrame({"missing": missing, "percent": percent})
    summary = summary[summary["missing"] > 0].sort_values("missing", ascending=False)
    return summary


def _save_figure(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(df: pd.DataFrame) -> dict:
    outputs = {}
    sns.set_theme(style="whitegrid")

    # Histogram cien
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["price"], bins=40, kde=True, ax=ax, color="#2a9d8f")
    ax.set_title("Rozdelenie cien notebookov (INR)")
    ax.set_xlabel("Cena [INR]")
    ax.set_ylabel("Počet záznamov")
    _save_figure(fig, "price_distribution.png")
    outputs["price_distribution"] = "reports/figures/price_distribution.png"

    # Log-histogram pre lepšiu škálu
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(np.log10(df["price"]), bins=40, kde=True, ax=ax, color="#264653")
    ax.set_title("Rozdelenie log10(ceny)")
    ax.set_xlabel("log10(Cena)")
    ax.set_ylabel("Počet záznamov")
    _save_figure(fig, "price_distribution_log10.png")
    outputs["price_distribution_log10"] = "reports/figures/price_distribution_log10.png"

    # Boxplot podľa značky (top 10)
    top_brands = df["brand"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df[df["brand"].isin(top_brands)],
        x="brand",
        y="price",
        ax=ax,
        color="#5e60ce",
    )
    ax.set_title("Cena podľa značky (top 10 podľa frekvencie)")
    ax.set_xlabel("Značka")
    ax.set_ylabel("Cena [INR]")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, "price_by_brand.png")
    outputs["price_by_brand"] = "reports/figures/price_by_brand.png"

    # Scatter cena vs. RAM
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="ram_gb", y="price", hue="storage_type", ax=ax, alpha=0.7)
    ax.set_title("Vzťah medzi RAM a cenou")
    ax.set_xlabel("RAM [GB]")
    ax.set_ylabel("Cena [INR]")
    _save_figure(fig, "price_vs_ram.png")
    outputs["price_vs_ram"] = "reports/figures/price_vs_ram.png"

    # Korelačná heatmapa numerických čŕt
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Korelácia"},
    )
    ax.set_title("Korelácia numerických čŕt")
    _save_figure(fig, "numeric_correlation_heatmap.png")
    outputs["numeric_correlation_heatmap"] = "reports/figures/numeric_correlation_heatmap.png"

    # Počet zariadení podľa typu úložiska
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="storage_type", order=df["storage_type"].value_counts().index, ax=ax)
    ax.set_title("Zastúpenie typov úložiska")
    ax.set_xlabel("Typ úložiska")
    ax.set_ylabel("Počet záznamov")
    _save_figure(fig, "storage_type_counts.png")
    outputs["storage_type_counts"] = "reports/figures/storage_type_counts.png"

    if "source" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="source", order=df["source"].value_counts().index, ax=ax)
        ax.set_title("Počet záznamov podľa zdroja")
        ax.set_xlabel("Zdroj")
        ax.set_ylabel("Počet záznamov")
        ax.tick_params(axis="x", rotation=45)
        _save_figure(fig, "records_by_source.png")
        outputs["records_by_source"] = "reports/figures/records_by_source.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="source",
            y="price",
            order=df["source"].value_counts().index,
            ax=ax,
            color="#5e60ce",
        )
        ax.set_title("Cena podľa zdroja datasetu")
        ax.set_xlabel("Zdroj")
        ax.set_ylabel("Cena [INR]")
        ax.tick_params(axis="x", rotation=45)
        _save_figure(fig, "price_by_source.png")
        outputs["price_by_source"] = "reports/figures/price_by_source.png"

    return outputs


def build_text_report(
    source_counts: Dict[str, int],
    df_clean: pd.DataFrame,
    figures: dict,
) -> str:
    summary_df = df_clean.drop(columns=["source"], errors="ignore")
    missing_summary = _describe_missing(summary_df)
    top_brands = summary_df["brand"].value_counts().head(10)
    price_stats = summary_df["price"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    cpu_vendor = add_engineered_features(summary_df)[
        "cpu_vendor"
    ].value_counts(normalize=True)
    storage_share = summary_df["storage_type"].value_counts(normalize=True)

    sources_text = "\n".join(f"  - {name}: {count}" for name, count in source_counts.items())

    text = dedent(
        f"""
        Dataset analysis report
        ----------------------
        Počet riadkov (spolu): {len(df_clean)}
        Zdroje:
{sources_text}

        Základné štatistiky cien (INR):
        {price_stats.to_string()}

        Najčastejšie značky (top 10):
        {top_brands.to_string()}

        Podiel CPU vendorov:
        {cpu_vendor.to_string(float_format=lambda x: f"{x*100:.1f}%")}

        Podiel typov úložísk:
        {storage_share.to_string(float_format=lambda x: f"{x*100:.1f}%")}
        """
    ).strip()

    if not missing_summary.empty:
        text += "\n\nChýbajúce hodnoty (top 10):\n"
        text += missing_summary.head(10).to_string()

    text += "\n\nGenerované grafy:\n"
    for name, path in figures.items():
        text += f"  - {name}: {path}\n"

    return text


def write_report(content: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    datasets = collect_clean_datasets()
    if not datasets:
        raise ValueError("Neboli nájdené žiadne datasety na analýzu.")

    frames = []
    for name, subset in datasets.items():
        temp = subset.copy()
        temp["source"] = name
        frames.append(temp)

    df_clean = pd.concat(frames, ignore_index=True)
    figures = generate_figures(df_clean)
    source_counts = {name: len(subset) for name, subset in datasets.items()}

    report_text = build_text_report(source_counts, df_clean, figures)
    write_report(report_text)

    print("Analýza dokončená.")
    print(report_text)


if __name__ == "__main__":
    main()

