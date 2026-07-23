"""Entraînement Keras, évaluation et suivi MLflow du TP auto-encodeur."""

from __future__ import annotations

import json
import logging
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from indusense.vision.anomaly import (
    calibrate_threshold,
    critical_analysis,
    image_metrics,
    pixel_error_maps,
    pixel_metrics,
    reconstruction_scores,
    save_confusion_matrix,
    save_heatmap_panel,
    save_history_plot,
    save_reconstruction_grid,
    save_score_histogram,
)
from indusense.vision.carbon import VisionEmissionsTracker
from indusense.vision.model import build_autoencoder, keras_api, model_description, model_summary_text
from indusense.vision.dataset import VisionPreparationConfig, load_model_input

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionTrainingConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    loss_name: str = "mse"
    latent_filters: int = 16
    threshold_percentile: float = 99.0
    early_stopping_patience: int = 5
    random_seed: int = 42

    def validate(self) -> None:
        if not 1 <= self.epochs <= 200:
            raise ValueError("Le nombre d'époques doit être compris entre 1 et 200.")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("La taille du lot doit être comprise entre 1 et 64.")
        if not 1e-6 <= self.learning_rate <= 0.1:
            raise ValueError("Le taux d'apprentissage doit être compris entre 1e-6 et 0,1.")
        if self.loss_name not in {"mse", "ssim"}:
            raise ValueError("La perte doit être MSE ou SSIM.")
        if not 1 <= self.latent_filters <= 128:
            raise ValueError("Le nombre de filtres latents doit être compris entre 1 et 128.")
        if not 50 <= self.threshold_percentile <= 100:
            raise ValueError("Le centile du seuil doit être compris entre 50 et 100.")


def train_vision_autoencoder(
    *,
    dataset_dir: Path,
    manifest: dict[str, Any],
    run_dir: Path,
    run_id: str,
    config: VisionTrainingConfig,
    mlflow_tracking_uri: str,
    experiment_name: str = "vision_autoencoder_b6",
) -> dict[str, Any]:
    """Exécute le TP de bout en bout et conserve tous les livrables."""

    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    heatmaps_dir = figures_dir / "heatmaps"
    figures_dir.mkdir(exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    keras = keras_api()
    keras.utils.set_random_seed(config.random_seed)

    preparation = VisionPreparationConfig(**manifest["config"])
    train_records = [item for item in manifest["images"] if item["split"] == "train" and not item["is_anomaly"]]
    validation_normal_records = [
        item for item in manifest["images"] if item["split"] == "validation" and not item["is_anomaly"]
    ]
    test_records = [item for item in manifest["images"] if item["split"] == "test"]
    if not train_records or not validation_normal_records or not test_records:
        raise ValueError("Les jeux d'entraînement sain, de validation saine et de test ne doivent pas être vides.")

    LOGGER.info(
        "Chargement vision : %s images saines d'entraînement, %s images saines de validation et %s images de test.",
        len(train_records),
        len(validation_normal_records),
        len(test_records),
    )
    validation_images = _load_images(dataset_dir, validation_normal_records, preparation)
    test_images = _load_images(dataset_dir, test_records, preparation)
    test_labels = np.asarray([int(item["is_anomaly"]) for item in test_records], dtype=np.int32)
    test_masks = _load_masks(dataset_dir, test_records, preparation.target_size)

    model = build_autoencoder(
        (preparation.target_size, preparation.target_size, 3),
        latent_filters=config.latent_filters,
        learning_rate=config.learning_rate,
        loss_name=config.loss_name,
    )
    architecture = model_description(model)
    summary_path = run_dir / "model_summary.txt"
    summary_path.write_text(model_summary_text(model) + "\n" + architecture["comment"] + "\n", encoding="utf-8")

    sequence = _training_sequence(keras, dataset_dir, train_records, preparation, config)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    # L'entraînement est le périmètre comparable entre deux variantes de modèle.
    with VisionEmissionsTracker(run_dir) as carbon_tracker:
        history_object = model.fit(
            sequence,
            validation_data=(validation_images, validation_images),
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )
    history = {key: [float(value) for value in values] for key, values in history_object.history.items()}
    model_path = run_dir / "autoencoder.keras"
    model.save(model_path)

    validation_reconstructions = np.asarray(model.predict(validation_images, batch_size=config.batch_size, verbose=0))
    test_reconstructions = np.asarray(model.predict(test_images, batch_size=config.batch_size, verbose=0))
    validation_scores = reconstruction_scores(validation_images, validation_reconstructions)
    test_scores = reconstruction_scores(test_images, test_reconstructions)
    threshold = calibrate_threshold(validation_scores, config.threshold_percentile)
    image_evaluation = image_metrics(test_labels, test_scores, threshold)
    error_maps = pixel_error_maps(test_images, test_reconstructions)
    pixel_evaluation = pixel_metrics(test_masks, error_maps)

    history_path = figures_dir / "learning_curve.png"
    histogram_path = figures_dir / "score_histogram.png"
    confusion_path = figures_dir / "confusion_matrix.png"
    reconstruction_path = figures_dir / "reconstructions.png"
    save_history_plot(history, history_path)
    save_score_histogram(test_scores[test_labels == 0], test_scores[test_labels == 1], threshold, histogram_path)
    save_confusion_matrix(image_evaluation["confusion_matrix"], confusion_path)
    save_reconstruction_grid(test_images, test_reconstructions, reconstruction_path)

    sample_results = []
    anomaly_indices = np.flatnonzero(test_labels == 1)
    selected_indices = anomaly_indices[np.argsort(test_scores[anomaly_indices])[-min(4, len(anomaly_indices)):]]
    for index in selected_indices:
        heatmap_path = heatmaps_dir / f"heatmap_{index:03d}.png"
        save_heatmap_panel(
            test_images[index],
            test_reconstructions[index],
            error_maps[index],
            test_masks[index],
            heatmap_path,
            title=f"{test_records[index]['label']} — {Path(test_records[index]['path']).name}",
        )
        sample_results.append(
            {
                "path": test_records[index]["path"],
                "label": test_records[index]["label"],
                "score": float(test_scores[index]),
                "predicted_anomaly": bool(test_scores[index] > threshold),
                "heatmap_artifact": heatmap_path.relative_to(run_dir).as_posix(),
            }
        )

    best_epoch = int(np.argmin(history.get("val_loss", history["loss"]))) + 1
    report = {
        "run_id": run_id,
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": manifest["version_id"],
        "dataset_hash": manifest["dataset_hash"],
        "config": asdict(config),
        "architecture": {**architecture, "summary_artifact": summary_path.relative_to(run_dir).as_posix()},
        "training": {
            "train_normal_images": len(train_records),
            "validation_normal_images": len(validation_normal_records),
            "test_images": len(test_records),
            "epochs_completed": len(history["loss"]),
            "best_epoch": best_epoch,
            "best_validation_loss": float(min(history.get("val_loss", history["loss"]))),
            "history": history,
            "augmentation": "à la volée, uniquement sur les images saines d'entraînement",
        },
        "carbon": carbon_tracker.result,
        "threshold": {
            "method": "centile des scores des images saines de validation",
            "percentile": config.threshold_percentile,
            "value": threshold,
            "calibration_images": len(validation_scores),
        },
        "metrics": {"image": image_evaluation, "pixel": pixel_evaluation},
        "score_summary": {
            "validation_normal_mean": float(np.mean(validation_scores)),
            "test_normal_mean": _optional_mean(test_scores[test_labels == 0]),
            "test_anomaly_mean": _optional_mean(test_scores[test_labels == 1]),
        },
        "samples": sample_results,
        "artifacts": {
            "model": model_path.relative_to(run_dir).as_posix(),
            "summary": summary_path.relative_to(run_dir).as_posix(),
            "learning_curve": history_path.relative_to(run_dir).as_posix(),
            "score_histogram": histogram_path.relative_to(run_dir).as_posix(),
            "confusion_matrix": confusion_path.relative_to(run_dir).as_posix(),
            "reconstructions": reconstruction_path.relative_to(run_dir).as_posix(),
        },
        "critical_analysis": critical_analysis(image_evaluation, architecture["compression_ratio"]),
        "reproducibility": {
            "python": platform.python_version(),
            "keras": keras.__version__,
            "backend": keras.backend.backend(),
            "random_seed": config.random_seed,
        },
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_run_id": None,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["mlflow_run_id"] = (
        _track_with_mlflow(report, run_dir, mlflow_tracking_uri, experiment_name) if mlflow_tracking_uri else None
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _training_sequence(
    keras: Any,
    dataset_dir: Path,
    records: list[dict[str, Any]],
    preparation: VisionPreparationConfig,
    config: VisionTrainingConfig,
) -> Any:
    class NormalImageSequence(keras.utils.PyDataset):
        def __init__(self) -> None:
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.indices = np.arange(len(records))
            self.epoch = 0
            self.rng = np.random.default_rng(config.random_seed)
            self.rng.shuffle(self.indices)

        def __len__(self) -> int:
            return int(np.ceil(len(records) / config.batch_size))

        def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
            selected = self.indices[batch_index * config.batch_size : (batch_index + 1) * config.batch_size]
            images = [
                load_model_input(
                    dataset_dir,
                    records[int(index)],
                    preparation,
                    augment=True,
                    seed=config.random_seed + self.epoch * len(records) + int(index),
                )
                for index in selected
            ]
            batch = np.stack(images).astype(np.float32)
            return batch, batch

        def on_epoch_end(self) -> None:
            self.epoch += 1
            self.rng.shuffle(self.indices)

    return NormalImageSequence()


def _load_images(
    dataset_dir: Path,
    records: list[dict[str, Any]],
    preparation: VisionPreparationConfig,
) -> np.ndarray:
    return np.stack([load_model_input(dataset_dir, record, preparation) for record in records]).astype(np.float32)


def _load_masks(dataset_dir: Path, records: list[dict[str, Any]], target_size: int) -> np.ndarray:
    masks = []
    for record in records:
        if not record.get("mask_path"):
            masks.append(np.zeros((target_size, target_size), dtype=np.float32))
            continue
        with Image.open(dataset_dir / record["mask_path"]) as source:
            source = ImageOps.exif_transpose(source).convert("L")
            contained = ImageOps.contain(source, (target_size, target_size), method=Image.Resampling.NEAREST)
            canvas = Image.new("L", (target_size, target_size), color=0)
            offset = ((target_size - contained.width) // 2, (target_size - contained.height) // 2)
            canvas.paste(contained, offset)
            masks.append((np.asarray(canvas, dtype=np.float32) / 255.0) > 0.5)
    return np.asarray(masks, dtype=np.float32)


def _optional_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _track_with_mlflow(report: dict[str, Any], run_dir: Path, tracking_uri: str, experiment_name: str) -> str | None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=report["run_id"]) as active_run:
            mlflow.log_params(
                {
                    **report["config"],
                    "dataset_version": report["dataset_version"],
                    "compression_ratio": report["architecture"]["compression_ratio"],
                    "parameter_count": report["architecture"]["parameter_count"],
                }
            )
            for step, loss in enumerate(report["training"]["history"]["loss"]):
                mlflow.log_metric("train_loss", loss, step=step)
            for step, loss in enumerate(report["training"]["history"].get("val_loss", [])):
                mlflow.log_metric("validation_loss", loss, step=step)
            image_result = report["metrics"]["image"]
            pixel_result = report["metrics"]["pixel"]
            metrics = {
                "threshold": report["threshold"]["value"],
                "image_precision": image_result["precision"],
                "image_recall": image_result["recall"],
                "image_f1": image_result["f1"],
            }
            if image_result["auroc"] is not None:
                metrics["image_auroc"] = image_result["auroc"]
            if pixel_result["auroc"] is not None:
                metrics["pixel_auroc"] = pixel_result["auroc"]
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(run_dir), artifact_path="vision_run")
            return active_run.info.run_id
    except Exception as error:  # noqa: BLE001 - le rapport local reste exploitable sans tracking.
        LOGGER.warning("Suivi MLflow de la vision indisponible : %s", error)
        return None
