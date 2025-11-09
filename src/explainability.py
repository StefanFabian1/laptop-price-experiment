from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.cleaning import build_combined_dataset
from src.data_prep import CAT_COLS, NUM_COLS, build_preprocessor
from src.features import add_engineered_features
from src.modeling import get_models


OUTPUT_DIR = Path("reports/explainability")


def _base_feature(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        return feature_name.split("__", 1)[1]
    if feature_name.startswith("cat__"):
        remainder = feature_name.split("__", 1)[1]
        for col in CAT_COLS:
            prefix = f"{col}_"
            if remainder.startswith(prefix):
                return col
        return remainder
    return feature_name


def compute_explainability(random_state: int = 42) -> Dict[str, Path]:
    df = build_combined_dataset()
    df = add_engineered_features(df)
    df["log_price"] = np.log(df["price"].clip(lower=1))

    X = df.drop(columns=["price", "log_price"])
    y = df["log_price"]

    preprocessor = build_preprocessor()
    model = get_models(random_state=random_state)["xgb"]
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X, y)

    transformer = pipe.named_steps["pre"]
    xgb_model = pipe.named_steps["model"]

    Xt = transformer.transform(X)
    if hasattr(Xt, "toarray"):
        Xt_dense = Xt.toarray()
    else:
        Xt_dense = Xt

    feature_names = transformer.get_feature_names_out()

    background_size = min(500, Xt_dense.shape[0])
    masker = shap.maskers.Independent(Xt_dense[:background_size])
    explainer = shap.Explainer(xgb_model.predict, masker, feature_names=feature_names)

    explain_size = min(1000, Xt_dense.shape[0])
    Xt_sample = Xt_dense[:explain_size]
    shap_values = explainer(Xt_sample).values

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values,
        Xt_sample,
        feature_names=feature_names,
        show=False,
        max_display=25,
    )
    summary_path = OUTPUT_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_csv_path = OUTPUT_DIR / "shap_feature_importance.csv"
    shap_df.to_csv(shap_csv_path, index=False)

    group_scores = (
        shap_df.assign(base_feature=lambda df_: df_["feature_name"].apply(_base_feature))
        .groupby("base_feature")["mean_abs_shap"]
        .sum()
        .sort_values(ascending=False)
    )
    top10 = group_scores.head(10)

    summary_lines = [
        "Explainability summary",
        "-----------------------",
        f"Počet vzoriek (dataset): {len(df)}",
        f"Počet vzoriek (SHAP sample): {explain_size}",
        f"Veľkosť background maskera: {background_size}",
        f"Model: XGBoost (random_state={random_state})",
        "",
        "Top 10 črť podľa |SHAP| (agregované na úrovni vstupných stĺpcov):",
    ]
    for feature, score in top10.items():
        summary_lines.append(f"  - {feature}: {score:.6f}")

    summary_lines.extend(
        [
            "",
            "Poznámka: SHAP hodnoty sú počítané na log(cena); interpretácia vo vzťahu k pôvodnej mene vyžaduje exponentáciu.",
            "Grafický súbor: shap_summary.png",
            "Tabuľka dôležitostí: shap_feature_importance.csv",
        ]
    )

    summary_txt_path = OUTPUT_DIR / "explainability_summary.txt"
    summary_txt_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "summary_png": summary_path,
        "importance_csv": shap_csv_path,
        "summary_txt": summary_txt_path,
    }


def main() -> None:
    outputs = compute_explainability()
    print("Explainability artefacts uložené:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

