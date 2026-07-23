"""Scores, seuils, métriques et figures pour la détection d'anomalies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def reconstruction_scores(images: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    _validate_image_batches(images, reconstructions)
    return np.mean(np.square(images - reconstructions), axis=(1, 2, 3), dtype=np.float64)


def pixel_error_maps(images: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    _validate_image_batches(images, reconstructions)
    return np.mean(np.square(images - reconstructions), axis=3, dtype=np.float64)


def calibrate_threshold(normal_validation_scores: np.ndarray, percentile: float = 99.0) -> float:
    values = np.asarray(normal_validation_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Les scores sains de validation doivent former un vecteur fini non vide.")
    if not 50.0 <= percentile <= 100.0:
        raise ValueError("Le centile doit être compris entre 50 et 100.")
    return float(np.percentile(values, percentile))


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": _safe_auc(labels, scores),
        "average_precision": _safe_average_precision(labels, scores),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pixel_metrics(masks: np.ndarray, error_maps: np.ndarray) -> dict[str, float | None]:
    truth = (np.asarray(masks).reshape(-1) > 0.5).astype(np.int8)
    scores = np.asarray(error_maps, dtype=np.float64).reshape(-1)
    return {
        "auroc": _safe_auc(truth, scores),
        "average_precision": _safe_average_precision(truth, scores),
    }


def save_history_plot(history: dict[str, Sequence[float]], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history.get("loss", []), label="train")
    axis.plot(history.get("val_loss", []), label="validation saine")
    axis.set(title="Courbe d'apprentissage", xlabel="Époque", ylabel="Perte")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, destination)


def save_score_histogram(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(normal_scores, bins=20, alpha=0.7, label="saines", color="#38bdf8")
    axis.hist(anomaly_scores, bins=20, alpha=0.7, label="défauts", color="#fb7185")
    axis.axvline(threshold, color="#facc15", linestyle="--", linewidth=2, label=f"seuil {threshold:.6f}")
    axis.set(title="Distribution des scores de reconstruction", xlabel="MSE par image", ylabel="Images")
    axis.grid(alpha=0.2)
    axis.legend()
    _save_figure(figure, destination)


def save_confusion_matrix(matrix: dict[str, int], destination: Path) -> None:
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row, column]), ha="center", va="center", color="black", fontsize=13)
    axis.set_xticks([0, 1], labels=["Saine", "Anomalie"])
    axis.set_yticks([0, 1], labels=["Saine", "Anomalie"])
    axis.set(xlabel="Prédiction", ylabel="Vérité", title="Matrice de confusion — test")
    figure.colorbar(image, ax=axis, fraction=0.046)
    _save_figure(figure, destination)


def save_reconstruction_grid(images: np.ndarray, reconstructions: np.ndarray, destination: Path, count: int = 4) -> None:
    count = min(count, len(images))
    figure, axes = plt.subplots(2, count, figsize=(3 * count, 6), squeeze=False)
    for index in range(count):
        axes[0, index].imshow(np.clip(images[index], 0, 1))
        axes[0, index].set_title("Original")
        axes[1, index].imshow(np.clip(reconstructions[index], 0, 1))
        axes[1, index].set_title("Reconstruction")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    _save_figure(figure, destination)


def save_heatmap_panel(
    original: np.ndarray,
    reconstruction: np.ndarray,
    error_map: np.ndarray,
    mask: np.ndarray,
    destination: Path,
    *,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title("Original")
    axes[1].imshow(np.clip(reconstruction, 0, 1))
    axes[1].set_title("Reconstruction")
    heat = axes[2].imshow(error_map, cmap="inferno")
    axes[2].set_title("Heatmap erreur")
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Masque de vérité terrain")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(heat, ax=axes[2], fraction=0.046)
    figure.suptitle(title)
    _save_figure(figure, destination)


def critical_analysis(metrics: dict[str, Any], compression_ratio: float) -> str:
    auroc = metrics.get("auroc")
    recall = metrics.get("recall", 0.0)
    matrix = metrics.get("confusion_matrix", {})
    false_negatives = int(matrix.get("fn", 0))
    false_positives = int(matrix.get("fp", 0))
    missed_defects = f"{false_negatives} défaut{'s' if false_negatives != 1 else ''} manqué{'s' if false_negatives != 1 else ''}"
    false_alerts = f"{false_positives} fausse{'s' if false_positives != 1 else ''} alerte{'s' if false_positives != 1 else ''}"
    auroc_text = f"{auroc:.3f}".replace(".", ",") if auroc is not None else "—"
    recall_text = f"{recall * 100:.1f}".replace(".", ",")
    compression_text = f"{compression_ratio:.2f}".replace(".", ",")
    performance = (
        "La séparation globale est bonne"
        if auroc is not None and auroc >= 0.9
        else "La séparation globale reste limitée"
    )
    return (
        f"{performance} (AUROC image = {auroc_text}). Le seuil calibré exclusivement sur les pièces saines "
        f"de validation atteint un rappel de {recall_text} %, avec {missed_defects} et {false_alerts}. "
        f"Le goulot compresse l'entrée par un facteur {compression_text} ; un ratio plus faible favoriserait "
        "une quasi-identité et pourrait faire disparaître le signal d'anomalie. Les cartes thermiques doivent "
        "être confrontées aux masques : une erreur "
        "sur le fond ou le contour peut augmenter le score sans localiser le défaut. Cette référence doit "
        "donc être comparée à une méthode fondée sur des caractéristiques préentraînées avant tout usage industriel."
    ) if auroc is not None else (
        "L'AUROC image n'est pas calculable, car le test ne contient pas les deux classes. "
        "Il faut conserver des images saines et défectueuses dans le test final."
    )


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(average_precision_score(labels, scores)) if np.unique(labels).size == 2 else None


def _validate_image_batches(images: np.ndarray, reconstructions: np.ndarray) -> None:
    if images.shape != reconstructions.shape or images.ndim != 4:
        raise ValueError("Les images et les reconstructions doivent avoir la même forme NHWC.")


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
