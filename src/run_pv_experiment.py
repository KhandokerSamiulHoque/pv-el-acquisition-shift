from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Paths: set ELPV_DATA to the folder that contains labels.csv and images/ from
# https://github.com/zae-bayern/elpv-dataset (src/elpv_dataset/data). Results go to ./results/run.
import os
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import bar_chart, confusion_chart, font


DATA_ROOT = Path(os.environ.get("ELPV_DATA", REPO / "data" / "elpv"))
RESULTS = REPO / "results" / "run"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 240,
    learning_rate: float = 0.035,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (features - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = 0.0
    v_b = 0.0
    positive_weight = len(labels) / max(2 * labels.sum(), 1)
    negative_weight = len(labels) / max(2 * (1 - labels).sum(), 1)
    sample_weight = np.where(labels == 1, positive_weight, negative_weight)
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        probability = sigmoid(x @ weights + bias)
        residual = (probability - labels) * sample_weight
        gradient_w = x.T @ residual / len(labels) + l2 * weights
        gradient_b = float(residual.mean())
        m_w = beta1 * m_w + (1 - beta1) * gradient_w
        v_w = beta2 * v_w + (1 - beta2) * (gradient_w**2)
        m_b = beta1 * m_b + (1 - beta1) * gradient_b
        v_b = beta2 * v_b + (1 - beta2) * (gradient_b**2)
        m_w_hat = m_w / (1 - beta1**epoch)
        v_w_hat = v_w / (1 - beta2**epoch)
        m_b_hat = m_b / (1 - beta1**epoch)
        v_b_hat = v_b / (1 - beta2**epoch)
        weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
        bias -= learning_rate * m_b_hat / (math.sqrt(v_b_hat) + 1e-8)
    return weights, bias, mean, scale


def predict_logistic(
    features: np.ndarray,
    model: tuple[np.ndarray, float, np.ndarray, np.ndarray],
) -> np.ndarray:
    weights, bias, mean, scale = model
    return sigmoid(((features - mean) / scale) @ weights + bias)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos = positives.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / max(sorted_labels.sum(), 1))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predicted = (scores >= 0.5).astype(int)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": average_precision(labels, scores),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def stratified_folds(labels: np.ndarray, splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(splits)]
    for value in [0, 1]:
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, splits)):
            fold_parts[fold].append(part)
    return [np.sort(np.concatenate(parts)) for parts in fold_parts]


def image_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    resized = np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize((96, 96), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    gy, gx = np.gradient(resized)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) + np.pi) % np.pi

    global_values = [
        resized.mean(),
        resized.std(),
        *np.quantile(resized, [0.10, 0.25, 0.50, 0.75, 0.90]),
        np.mean(resized < 0.20),
        np.mean(resized < 0.40),
        np.mean(resized > 0.80),
        magnitude.mean(),
        magnitude.std(),
        np.quantile(magnitude, 0.90),
        resized[24:72, 24:72].mean()
        - np.concatenate(
            [resized[:12].ravel(), resized[-12:].ravel(), resized[:, :12].ravel(), resized[:, -12:].ravel()]
        ).mean(),
    ]
    names = [
        "intensity_mean",
        "intensity_std",
        "intensity_q10",
        "intensity_q25",
        "intensity_median",
        "intensity_q75",
        "intensity_q90",
        "dark_fraction_020",
        "dark_fraction_040",
        "bright_fraction_080",
        "gradient_mean",
        "gradient_std",
        "gradient_q90",
        "center_minus_border_intensity",
    ]
    grid_values: list[float] = []
    for row in range(4):
        for col in range(4):
            ys = slice(row * 24, (row + 1) * 24)
            xs = slice(col * 24, (col + 1) * 24)
            for metric_name, value in (
                ("mean", resized[ys, xs].mean()),
                ("std", resized[ys, xs].std()),
                ("gradient", magnitude[ys, xs].mean()),
            ):
                grid_values.append(float(value))
                names.append(f"grid_r{row + 1}c{col + 1}_{metric_name}")
    interpretable = np.asarray(global_values + grid_values, dtype=np.float64)

    hog_values: list[float] = []
    bins = np.linspace(0, np.pi, 10)
    for row in range(6):
        for col in range(6):
            ys = slice(row * 16, (row + 1) * 16)
            xs = slice(col * 16, (col + 1) * 16)
            histogram, _ = np.histogram(
                orientation[ys, xs],
                bins=bins,
                weights=magnitude[ys, xs],
            )
            histogram = histogram / max(np.linalg.norm(histogram), 1e-12)
            hog_values.extend(histogram.tolist())
    return interpretable, np.asarray(hog_values), names


def load_dataset():
    records = pd.read_csv(
        DATA_ROOT / "labels.csv",
        sep=r"\s+",
        header=None,
        names=["path", "probability", "cell_type"],
    )
    interpretable_rows, hog_rows, thumbnails = [], [], []
    names = []
    for relative_path in records["path"]:
        image = np.asarray(Image.open(DATA_ROOT / relative_path).convert("L"))
        interpretable, hog_values, names = image_features(image)
        interpretable_rows.append(interpretable)
        hog_rows.append(hog_values)
        thumbnails.append(Image.fromarray(image).resize((180, 180)))
    records["any_defect"] = (records["probability"] > 0).astype(int)
    records["high_confidence_defect"] = (records["probability"] >= (2 / 3)).astype(int)
    return records, np.vstack(interpretable_rows), np.vstack(hog_rows), names, thumbnails


def evaluate_task(
    task: str,
    labels: np.ndarray,
    cell_types: np.ndarray,
    feature_sets: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    folds = stratified_folds(labels, 5, RANDOM_SEED)
    rows = []
    oof_by_model = {name: np.zeros(len(labels)) for name in feature_sets}
    all_indices = np.arange(len(labels))
    for model_name, features in feature_sets.items():
        for fold_number, test_indices in enumerate(folds, start=1):
            train_indices = np.setdiff1d(all_indices, test_indices)
            model = fit_logistic(features[train_indices], labels[train_indices])
            scores = predict_logistic(features[test_indices], model)
            oof_by_model[model_name][test_indices] = scores
            row = metrics(labels[test_indices], scores)
            row.update({"task": task, "model": model_name, "fold": fold_number})
            rows.append(row)
    cv = pd.DataFrame(rows)
    summary = cv.groupby("model")[["f1", "roc_auc", "pr_auc"]].mean()
    best_model = summary["f1"].idxmax()
    best_scores = oof_by_model[best_model]
    overall = metrics(labels, best_scores)
    subgroup_rows = []
    for subgroup in sorted(set(cell_types)):
        mask = cell_types == subgroup
        subgroup_rows.append(
            {
                "subgroup": subgroup,
                "n": int(mask.sum()),
                "positive_rate": float(labels[mask].mean()),
                **metrics(labels[mask], best_scores[mask]),
            }
        )
    payload = {
        "task": task,
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
        "negative_n": int((1 - labels).sum()),
        "best_model": best_model,
        "oof_metrics": overall,
        "confusion_matrix": [
            [overall["true_negative"], overall["false_positive"]],
            [overall["false_negative"], overall["true_positive"]],
        ],
        "subgroups": subgroup_rows,
    }
    return cv, payload


def sample_grid(records: pd.DataFrame, thumbnails: list[Image.Image]) -> None:
    canvas = Image.new("RGB", (900, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), "Representative ELPV cells by annotated defect probability", fill="#17365D", font=font(30, True))
    for col, probability in enumerate([0.0, 1 / 3, 2 / 3, 1.0]):
        indices = np.flatnonzero(np.isclose(records["probability"].to_numpy(), probability))[:3]
        draw.text((40 + col * 215, 75), f"p = {probability:.2f}", fill="#24364B", font=font(22, True))
        for row, index in enumerate(indices):
            canvas.paste(thumbnails[index].convert("RGB"), (35 + col * 215, 115 + row * 195))
            draw.text((40 + col * 215, 295 + row * 195), records.iloc[index]["cell_type"], fill="#24364B", font=font(18))
    canvas.save(RESULTS / "pv_example_cells.png")


def main() -> None:
    records, interpretable, hog_features, feature_names, thumbnails = load_dataset()
    feature_sets = {
        "Interpretable": interpretable,
        "HOG": hog_features,
        "Combined": np.hstack([interpretable, hog_features]),
    }
    cv_frames = []
    payloads = []
    for task, label_column in (
        ("Any annotated defect", "any_defect"),
        ("High-confidence defect", "high_confidence_defect"),
    ):
        cv, payload = evaluate_task(
            task,
            records[label_column].to_numpy(),
            records["cell_type"].to_numpy(),
            feature_sets,
        )
        cv_frames.append(cv)
        payloads.append(payload)
    cv_results = pd.concat(cv_frames, ignore_index=True)
    cv_results.to_csv(RESULTS / "pv_cv_results.csv", index=False)
    summary = (
        cv_results.groupby(["task", "model"])
        .agg(
            n_folds=("fold", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "pv_summary_results.csv", index=False)

    any_defect = records["any_defect"].to_numpy()
    full_model = fit_logistic(interpretable, any_defect)
    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": full_model[0],
            "absolute_coefficient": np.abs(full_model[0]),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_table.to_csv(RESULTS / "pv_interpretable_coefficients.csv", index=False)

    primary = summary[summary["task"] == "Any annotated defect"]
    groups = primary["model"].tolist()
    bar_chart(
        RESULTS / "pv_model_comparison.png",
        "PV defect-screening model comparison",
        groups,
        {
            "F1": primary["f1_mean"].tolist(),
            "ROC AUC": primary["roc_auc_mean"].tolist(),
            "PR AUC": primary["pr_auc_mean"].tolist(),
        },
        minimum=0.45,
    )
    any_payload = next(item for item in payloads if item["task"] == "Any annotated defect")
    confusion_chart(
        RESULTS / "pv_confusion_matrix.png",
        "Out-of-fold confusion matrix",
        any_payload["confusion_matrix"],
    )
    sample_grid(records, thumbnails)

    payload = {
        "random_seed": RANDOM_SEED,
        "dataset_summary": {
            "dataset": "ELPV",
            "records": int(len(records)),
            "modules_reported_by_source": 44,
            "image_shape_reported_by_source": "300x300 grayscale",
            "cell_type_counts": {key: int(value) for key, value in records["cell_type"].value_counts().items()},
            "probability_counts": {str(key): int(value) for key, value in records["probability"].value_counts().sort_index().items()},
            "feature_dimensions": {name: int(values.shape[1]) for name, values in feature_sets.items()},
        },
        "tasks": payloads,
        "limitations": [
            "The public labels file does not expose module identifiers, so module-grouped cross-validation could not be performed.",
            "Defect probability is an expert annotation, not a direct measurement of power loss or field failure.",
            "The experiment evaluates a public benchmark and does not validate deployment on production-line or field-acquired images.",
        ],
    }
    (RESULTS / "pv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payloads, indent=2))


if __name__ == "__main__":
    main()
