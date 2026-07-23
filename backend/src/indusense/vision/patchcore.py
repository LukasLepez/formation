"""Implémentation PatchCore compacte pour la détection d'anomalies visuelles.

Le modèle n'est pas entraîné par descente de gradient : il extrait des patchs avec
un ResNet-18 ImageNet pré-entraîné, construit une banque de patchs sains puis
mesure la distance au voisin sain le plus proche.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from indusense.vision.anomaly import (
    calibrate_threshold,
    critical_analysis,
    image_metrics,
    pixel_metrics,
    save_confusion_matrix,
    save_heatmap_panel,
    save_reconstruction_grid,
    save_score_histogram,
)
from indusense.vision_dataset import VisionPreparationConfig, load_model_input

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchCoreConfig:
    """Paramètres de la banque de mémoire PatchCore."""

    backbone: str = "resnet18"
    coreset_ratio: float = 0.05
    max_memory_patches: int = 1024
    candidate_patches: int = 10000
    batch_size: int = 8
    threshold_percentile: float = 99.0
    random_seed: int = 42

    def validate(self) -> None:
        if self.backbone != "resnet18":
            raise ValueError("Le backbone PatchCore disponible est resnet18.")
        if not 0.001 <= self.coreset_ratio <= 1.0:
            raise ValueError("Le ratio de coreset doit être compris entre 0,001 et 1.")
        if not 32 <= self.max_memory_patches <= 20_000:
            raise ValueError("La taille de mémoire doit être comprise entre 32 et 20 000 patchs.")
        if not 512 <= self.candidate_patches <= 100_000:
            raise ValueError("Le nombre de candidats doit être compris entre 512 et 100 000.")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("La taille du lot doit être comprise entre 1 et 64.")
        if not 50 <= self.threshold_percentile <= 100:
            raise ValueError("Le centile du seuil doit être compris entre 50 et 100.")


def train_patchcore(
    *,
    dataset_dir: Path,
    manifest: dict[str, Any],
    run_dir: Path,
    run_id: str,
    config: PatchCoreConfig,
    mlflow_tracking_uri: str,
    experiment_name: str = "vision_patchcore_b6",
) -> dict[str, Any]:
    """Construit la mémoire saine et évalue PatchCore sur le test final."""

    config.validate()
    torch, extractor, device = _load_extractor()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    figures_dir = run_dir / "figures"
    heatmaps_dir = figures_dir / "heatmaps"
    figures_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)
    preparation = VisionPreparationConfig(**manifest["config"])
    train_records = [item for item in manifest["images"] if item["split"] == "train" and not item["is_anomaly"]]
    validation_records = [item for item in manifest["images"] if item["split"] == "validation" and not item["is_anomaly"]]
    test_records = [item for item in manifest["images"] if item["split"] == "test"]
    if not train_records or not validation_records or not test_records:
        raise ValueError("Les jeux sain d'entraînement, sain de validation et de test ne doivent pas être vides.")

    LOGGER.info("PatchCore : extraction des patchs de %s images saines.", len(train_records))
    train_images = _load_images(dataset_dir, train_records, preparation)
    train_features = _extract_features(torch, extractor, device, train_images, config.batch_size)
    all_patches = train_features.reshape(-1, train_features.shape[-1])
    memory = _build_coreset(
        torch,
        all_patches,
        ratio=config.coreset_ratio,
        max_memory_patches=config.max_memory_patches,
        candidate_patches=config.candidate_patches,
        seed=config.random_seed,
    )
    memory_path = run_dir / "patchcore_memory.npz"
    np.savez_compressed(memory_path, memory=memory.cpu().numpy())

    validation_images = _load_images(dataset_dir, validation_records, preparation)
    test_images = _load_images(dataset_dir, test_records, preparation)
    validation_scores, _ = _score_images(torch, extractor, device, validation_images, memory, config.batch_size)
    test_scores, test_maps = _score_images(torch, extractor, device, test_images, memory, config.batch_size)
    test_labels = np.asarray([int(item["is_anomaly"]) for item in test_records], dtype=np.int32)
    test_masks = _load_masks(dataset_dir, test_records, preparation.target_size)
    threshold = calibrate_threshold(validation_scores, config.threshold_percentile)
    image_evaluation = image_metrics(test_labels, test_scores, threshold)
    pixel_evaluation = pixel_metrics(test_masks, test_maps)

    summary_path = run_dir / "model_summary.txt"
    summary_path.write_text(
        "PatchCore\n\n"
        "Backbone : ResNet-18 ImageNet pré-entraîné (features layer1 + layer2).\n"
        f"Patchs initiaux : {len(all_patches):,}\n"
        f"Candidats du coreset : {min(len(all_patches), config.candidate_patches):,}\n"
        f"Banque mémoire finale : {len(memory):,} patchs de dimension {memory.shape[1]}.\n"
        "Score : distance euclidienne du patch au voisin sain le plus proche ; score image = maximum des patchs.\n",
        encoding="utf-8",
    )
    _save_patchcore_overview(figures_dir / "learning_curve.png", len(all_patches), len(memory), config)
    save_score_histogram(test_scores[test_labels == 0], test_scores[test_labels == 1], threshold, figures_dir / "score_histogram.png")
    save_confusion_matrix(image_evaluation["confusion_matrix"], figures_dir / "confusion_matrix.png")
    _save_patchcore_examples(test_images, test_maps, figures_dir / "reconstructions.png")

    samples = []
    anomaly_indices = np.flatnonzero(test_labels == 1)
    selected = anomaly_indices[np.argsort(test_scores[anomaly_indices])[-min(4, len(anomaly_indices)):]]
    for index in selected:
        heatmap_path = heatmaps_dir / f"heatmap_{index:03d}.png"
        save_heatmap_panel(
            test_images[index], test_images[index], test_maps[index], test_masks[index], heatmap_path,
            title=f"{test_records[index]['label']} — {Path(test_records[index]['path']).name}",
        )
        samples.append({"path": test_records[index]["path"], "label": test_records[index]["label"], "score": float(test_scores[index]), "predicted_anomaly": bool(test_scores[index] > threshold), "heatmap_artifact": heatmap_path.relative_to(run_dir).as_posix()})

    architecture = {
        "input_shape": [preparation.target_size, preparation.target_size, 3],
        "latent_shape": [int(test_maps.shape[1]), int(test_maps.shape[2]), int(memory.shape[1])],
        "input_values": preparation.target_size * preparation.target_size * 3,
        "latent_values": int(memory.shape[1]),
        "compression_ratio": 0.0,
        "parameter_count": 0,
        "comment": f"PatchCore utilise une banque de {len(memory):,} patchs sains extraits par ResNet-18 pré-entraîné ; il n'entraîne pas de décodeur.",
        "summary_artifact": summary_path.relative_to(run_dir).as_posix(),
    }
    report = {
        "run_id": run_id, "status": "success", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "patchcore", "dataset_version": manifest["version_id"], "dataset_hash": manifest["dataset_hash"],
        "config": asdict(config), "architecture": architecture,
        "training": {"train_normal_images": len(train_records), "validation_normal_images": len(validation_records), "test_images": len(test_records), "epochs_completed": 0, "best_epoch": 0, "best_validation_loss": None, "history": {"loss": [], "val_loss": []}, "augmentation": "aucune : PatchCore construit la banque à partir des images saines originales"},
        "threshold": {"method": "centile des scores des images saines de validation", "percentile": config.threshold_percentile, "value": threshold, "calibration_images": len(validation_scores)},
        "metrics": {"image": image_evaluation, "pixel": pixel_evaluation},
        "score_summary": {"validation_normal_mean": float(np.mean(validation_scores)), "test_normal_mean": _mean_or_none(test_scores[test_labels == 0]), "test_anomaly_mean": _mean_or_none(test_scores[test_labels == 1])},
        "samples": samples,
        "artifacts": {"model": memory_path.relative_to(run_dir).as_posix(), "summary": summary_path.relative_to(run_dir).as_posix(), "learning_curve": "figures/learning_curve.png", "score_histogram": "figures/score_histogram.png", "confusion_matrix": "figures/confusion_matrix.png", "reconstructions": "figures/reconstructions.png"},
        "critical_analysis": critical_analysis(image_evaluation, 1.0),
        "reproducibility": {"backbone": config.backbone, "device": str(device), "random_seed": config.random_seed},
        "mlflow_tracking_uri": mlflow_tracking_uri, "mlflow_run_id": None,
    }
    report["mlflow_run_id"] = _track_patchcore(report, run_dir, mlflow_tracking_uri, experiment_name)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _load_extractor() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise RuntimeError("PatchCore requiert torch et torchvision. Lancez `uv sync` dans backend.") from error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()

    class Extractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1, self.layer2 = backbone.layer1, backbone.layer2

        def forward(self, values: Any) -> Any:
            values = self.stem(values)
            first = self.layer1(values)
            second = self.layer2(first)
            first = torch.nn.functional.adaptive_avg_pool2d(first, second.shape[-2:])
            return torch.cat((first, second), dim=1)

    return torch, Extractor().to(device).eval(), device


def _extract_features(torch: Any, extractor: Any, device: Any, images: np.ndarray, batch_size: int) -> Any:
    outputs = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
            features = extractor((batch - mean) / std).permute(0, 2, 3, 1).reshape(len(batch), -1, 192)
            outputs.append(features.cpu())
    return torch.cat(outputs)


def _build_coreset(torch: Any, patches: Any, *, ratio: float, max_memory_patches: int, candidate_patches: int, seed: int) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = patches[torch.randperm(len(patches), generator=generator)[:min(len(patches), candidate_patches)]].float()
    count = min(max_memory_patches, max(1, round(len(patches) * ratio)), len(candidates))
    if count == len(candidates):
        return candidates
    # Sélection greedy k-center : un coreset représentatif, au prix d'un calcul plus long.
    selected = torch.empty(count, dtype=torch.long)
    selected[0] = int(torch.randint(len(candidates), (1,), generator=generator).item())
    minimum = torch.sum((candidates - candidates[selected[0]]) ** 2, dim=1)
    for index in range(1, count):
        selected[index] = int(torch.argmax(minimum).item())
        distances = torch.sum((candidates - candidates[selected[index]]) ** 2, dim=1)
        minimum = torch.minimum(minimum, distances)
    return candidates[selected]


def _score_images(torch: Any, extractor: Any, device: Any, images: np.ndarray, memory: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    features = _extract_features(torch, extractor, device, images, batch_size).float()
    memory = memory.float()
    scores, maps = [], []
    for image_features in features:
        chunks = []
        for start in range(0, len(image_features), 256):
            chunks.append(torch.cdist(image_features[start:start + 256], memory).min(dim=1).values)
        patch_scores = torch.cat(chunks)
        side = int(np.sqrt(len(patch_scores)))
        patch_map = patch_scores.reshape(1, 1, side, side)
        resized = torch.nn.functional.interpolate(patch_map, size=images.shape[1:3], mode="bilinear", align_corners=False)[0, 0]
        scores.append(float(patch_scores.max().item()))
        maps.append(resized.numpy())
    return np.asarray(scores, dtype=np.float64), np.asarray(maps, dtype=np.float64)


def _load_images(dataset_dir: Path, records: list[dict[str, Any]], preparation: VisionPreparationConfig) -> np.ndarray:
    return np.stack([load_model_input(dataset_dir, record, preparation) for record in records]).astype(np.float32)


def _load_masks(dataset_dir: Path, records: list[dict[str, Any]], target_size: int) -> np.ndarray:
    masks = []
    for record in records:
        if not record.get("mask_path"):
            masks.append(np.zeros((target_size, target_size), dtype=np.float32)); continue
        with Image.open(dataset_dir / record["mask_path"]) as source:
            source = source.convert("L").resize((target_size, target_size), Image.Resampling.NEAREST)
            masks.append((np.asarray(source, dtype=np.float32) / 255.0) > 0.5)
    return np.asarray(masks, dtype=np.float32)


def _save_patchcore_overview(destination: Path, all_patches: int, memory_patches: int, config: PatchCoreConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(["patchs sains", "candidats", "mémoire"], [all_patches, min(all_patches, config.candidate_patches), memory_patches], color=["#38bdf8", "#facc15", "#a78bfa"])
    axis.set(title="Construction de la mémoire PatchCore", ylabel="Nombre de patchs")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(destination, dpi=150, bbox_inches="tight"); plt.close(figure)


def _save_patchcore_examples(images: np.ndarray, maps: np.ndarray, destination: Path) -> None:
    # Les reconstructions n'existent pas dans PatchCore : l'interface conserve cette tuile comme diagnostic image/carte.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    count = min(4, len(images)); figure, axes = plt.subplots(2, count, figsize=(3 * count, 6), squeeze=False)
    for index in range(count):
        axes[0, index].imshow(images[index]); axes[0, index].set_title("Image")
        axes[1, index].imshow(maps[index], cmap="inferno"); axes[1, index].set_title("Score PatchCore")
        axes[0, index].axis("off"); axes[1, index].axis("off")
    figure.tight_layout(); figure.savefig(destination, dpi=150, bbox_inches="tight"); plt.close(figure)


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _track_patchcore(report: dict[str, Any], run_dir: Path, tracking_uri: str, experiment_name: str) -> str | None:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri); mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=report["run_id"]) as active_run:
            mlflow.log_params({**report["config"], "dataset_version": report["dataset_version"], "model_type": "patchcore"})
            metrics = report["metrics"]
            mlflow.log_metrics({"threshold": report["threshold"]["value"], "image_auroc": metrics["image"]["auroc"] or 0.0, "image_f1": metrics["image"]["f1"], "pixel_auroc": metrics["pixel"]["auroc"] or 0.0})
            mlflow.log_artifacts(str(run_dir), artifact_path="vision_run")
            return active_run.info.run_id
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Suivi MLflow PatchCore indisponible : %s", error)
        return None
