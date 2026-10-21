from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from plot_utils import bar_chart
from run_pv_experiment import (
    DATA_ROOT,
    fit_logistic,
    image_features,
    metrics,
    predict_logistic,
    stratified_folds,
)


RESULTS = Path(__file__).resolve().parents[1] / "results" / "run"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 20260727
N_FOLDS = 5


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return result


def shift_image(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    values = image.astype(np.float64)
    if condition == "Reference":
        shifted = values
    elif condition == "Underexposure":
        shifted = values * 0.70
    elif condition == "Overexposure":
        shifted = values * 1.15 + 20.0
    elif condition == "Low contrast":
        shifted = 128.0 + (values - 128.0) * 0.65
    elif condition == "Gaussian noise":
        shifted = values + rng.normal(0.0, 12.0, size=values.shape)
    elif condition == "Gaussian blur":
        return np.asarray(
            Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=1.25)),
            dtype=np.uint8,
        )
    else:
        raise ValueError(condition)
    return np.clip(np.rint(shifted), 0, 255).astype(np.uint8)


def extract_condition_features(records: pd.DataFrame, condition: str) -> np.ndarray:
    cache = RESULTS / f"features_{condition.lower().replace(' ', '_')}.npz"
    if cache.exists():
        return np.load(cache)["features"]
    rows = []
    for index, relative_path in enumerate(records["path"]):
        image = np.asarray(Image.open(DATA_ROOT / relative_path).convert("L"), dtype=np.uint8)
        rng = np.random.default_rng(RANDOM_SEED + index * 1009)
        shifted = shift_image(image, condition, rng)
        interpretable, hog, _ = image_features(shifted)
        rows.append(np.concatenate([interpretable, hog]))
    features = np.vstack(rows)
    np.savez_compressed(cache, features=features)
    return features


def condition_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    base = metrics(labels, scores)
    confident = (scores < 0.40) | (scores > 0.60)
    confident_predictions = (scores[confident] >= 0.50).astype(int)
    base.update(
        {
            "brier_score": float(np.mean((scores - labels) ** 2)),
            "ece_10_bin": expected_calibration_error(labels, scores),
            "selective_coverage_040_060": float(confident.mean()),
            "selective_accuracy_040_060": float(np.mean(confident_predictions == labels[confident])),
        }
    )
    return base


def main() -> None:
    records = pd.read_csv(
        DATA_ROOT / "labels.csv",
        sep=r"\s+",
        header=None,
        names=["path", "probability", "cell_type"],
    )
    labels = (records["probability"].to_numpy() > 0).astype(int)
    conditions = [
        "Reference",
        "Underexposure",
        "Overexposure",
        "Low contrast",
        "Gaussian noise",
        "Gaussian blur",
    ]
    feature_sets = {condition: extract_condition_features(records, condition) for condition in conditions}
    folds = stratified_folds(labels, N_FOLDS, RANDOM_SEED)
    all_indices = np.arange(len(labels))
    scores = {condition: np.zeros(len(labels), dtype=np.float64) for condition in conditions}

    for test_indices in folds:
        train_indices = np.setdiff1d(all_indices, test_indices)
        model = fit_logistic(
            feature_sets["Reference"][train_indices],
            labels[train_indices],
            epochs=240,
            learning_rate=0.035,
            l2=1e-3,
        )
        for condition in conditions:
            scores[condition][test_indices] = predict_logistic(feature_sets[condition][test_indices], model)

    rows = []
    subgroup_rows = []
    for condition in conditions:
        row = {"condition": condition, **condition_metrics(labels, scores[condition])}
        rows.append(row)
        for cell_type in sorted(records["cell_type"].unique()):
            mask = records["cell_type"].to_numpy() == cell_type
            subgroup_rows.append(
                {
                    "condition": condition,
                    "cell_type": cell_type,
                    "n": int(mask.sum()),
                    **condition_metrics(labels[mask], scores[condition][mask]),
                }
            )

    summary = pd.DataFrame(rows)
    subgroups = pd.DataFrame(subgroup_rows)
    summary.to_csv(RESULTS / "pv_extension_shift_results.csv", index=False)
    subgroups.to_csv(RESULTS / "pv_extension_subgroup_results.csv", index=False)
    for condition in conditions:
        np.save(RESULTS / f"oof_scores_{condition.lower().replace(' ', '_')}.npy", scores[condition])

    reference_f1 = float(summary.loc[summary["condition"] == "Reference", "f1"].iloc[0])
    summary["absolute_f1_change_from_reference"] = summary["f1"] - reference_f1
    summary.to_csv(RESULTS / "pv_extension_shift_results.csv", index=False)

    bar_chart(
        RESULTS / "pv_extension_shift_robustness.png",
        "PV screening under deterministic acquisition shifts",
        summary["condition"].tolist(),
        {
            "F1": summary["f1"].tolist(),
            "ROC AUC": summary["roc_auc"].tolist(),
            "Selective accuracy": summary["selective_accuracy_040_060"].tolist(),
        },
        minimum=0.35,
    )
    bar_chart(
        RESULTS / "pv_extension_calibration.png",
        "Calibration error increases under acquisition shift",
        summary["condition"].tolist(),
        {
            "Brier score": summary["brier_score"].tolist(),
            "10-bin ECE": summary["ece_10_bin"].tolist(),
        },
        minimum=0.0,
    )

    payload = {
        "random_seed": RANDOM_SEED,
        "dataset": "ELPV",
        "records": int(len(records)),
        "target": "any annotated defect (probability > 0)",
        "validation": "Stratified five-fold out-of-fold evaluation; models trained only on unshifted reference images.",
        "feature_representation": "386 variables: 62 interpretable intensity/spatial/gradient features plus 324 HOG variables.",
        "conditions": {
            "Reference": "Original grayscale images.",
            "Underexposure": "Pixel intensities multiplied by 0.70.",
            "Overexposure": "Pixel intensities multiplied by 1.15 and increased by 20, then clipped.",
            "Low contrast": "Contrast about 128 reduced to 65 percent.",
            "Gaussian noise": "Independent deterministic Gaussian noise with standard deviation 12 intensity units.",
            "Gaussian blur": "PIL Gaussian blur with radius 1.25 pixels.",
        },
        "results": rows,
        "limitations": [
            "The transformations are controlled acquisition-shift stress tests, not an external dataset.",
            "The public labels file does not expose module identifiers, so module-grouped validation remains future work.",
            "Labels represent expert-assigned defect probabilities rather than measured electrical loss or field failure.",
            "The study establishes external-validation readiness and failure modes; it does not claim external or physical validation has been completed.",
            "The 0.40-0.60 review band is an illustrative triage rule and was not optimized for a deployment cost function.",
        ],
    }
    (RESULTS / "pv_extension_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(subgroups[["condition", "cell_type", "n", "f1", "roc_auc", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
