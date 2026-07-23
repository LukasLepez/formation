"""Préparation reproductible d'un jeu de données MVTec pour la détection d'anomalies."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class VisionPreparationConfig:
    """Paramètres qui déterminent intégralement une version du jeu de données."""

    target_size: int = 256
    validation_ratio: float = 0.2
    defect_validation_ratio: float = 0.3
    random_seed: int = 42
    padding_value: int = 0
    interpolation: str = "bilinear"

    def validate(self) -> None:
        if not 64 <= self.target_size <= 1024:
            raise ValueError("La taille cible doit être comprise entre 64 et 1024 pixels.")
        if not 0.05 <= self.validation_ratio <= 0.4:
            raise ValueError("La part de validation doit être comprise entre 5 % et 40 %.")
        if not 0.1 <= self.defect_validation_ratio <= 0.5:
            raise ValueError("La part des défauts réservés à la validation doit être comprise entre 10 % et 50 %.")
        if not 0 <= self.padding_value <= 255:
            raise ValueError("La valeur de remplissage doit être comprise entre 0 et 255.")
        if self.interpolation not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError("Méthode d'interpolation inconnue.")


def prepare_vision_dataset(
    dataset_dir: Path,
    artifacts_dir: Path,
    config: VisionPreparationConfig,
) -> dict[str, Any]:
    """Construit et conserve le manifeste sans modifier ni dupliquer les images."""

    config.validate()
    dataset_dir = dataset_dir.resolve()
    _validate_mvtec_structure(dataset_dir)

    train_good_paths = sorted((dataset_dir / "train" / "good").glob("*.png"))
    ordered_for_split = sorted(
        train_good_paths,
        key=lambda path: _stable_split_key(path.relative_to(dataset_dir).as_posix(), config.random_seed),
    )
    validation_count = max(1, round(len(ordered_for_split) * config.validation_ratio))
    validation_paths = set(ordered_for_split[:validation_count])

    records: list[dict[str, Any]] = []
    for path in train_good_paths:
        records.append(
            _image_record(
                dataset_dir,
                path,
                split="validation" if path in validation_paths else "train",
                label="good",
                is_anomaly=False,
            )
        )

    test_dir = dataset_dir / "test"
    for class_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        is_anomaly = class_dir.name != "good"
        class_paths = sorted(class_dir.glob("*.png"))
        validation_defects: set[Path] = set()
        if is_anomaly:
            ordered_defects = sorted(
                class_paths,
                key=lambda path: _stable_split_key(path.relative_to(dataset_dir).as_posix(), config.random_seed),
            )
            defect_validation_count = max(1, round(len(ordered_defects) * config.defect_validation_ratio))
            validation_defects = set(ordered_defects[:defect_validation_count])
        for path in class_paths:
            mask_path = _matching_mask(dataset_dir, class_dir.name, path.stem) if is_anomaly else None
            records.append(
                _image_record(
                    dataset_dir,
                    path,
                    split="validation" if path in validation_defects else "test",
                    label=class_dir.name,
                    is_anomaly=is_anomaly,
                    mask_path=mask_path,
                )
            )

    leakage = _leakage_report(records)
    if leakage["cross_split_duplicates"]:
        raise ValueError(
            "Fuite de données détectée : une même image apparaît dans plusieurs jeux. "
            "Consultez les empreintes SHA-256 avant de poursuivre."
        )

    train_records = [record for record in records if record["split"] == "train"]
    normalization = _normalization_statistics(dataset_dir, train_records, config)
    split_counts = _count_by(records, "split")
    class_counts = _count_by(records, "label")

    version_payload = {
        "dataset": "mvtec-ad/bottle",
        "config": asdict(config),
        "images": [
            {
                "path": record["path"],
                "split": record["split"],
                "label": record["label"],
                "sha256": record["sha256"],
                "mask_path": record["mask_path"],
                "mask_sha256": record["mask_sha256"],
            }
            for record in records
        ],
    }
    dataset_hash = hashlib.sha256(_canonical_json(version_payload).encode("utf-8")).hexdigest()
    version_id = f"bottle-{dataset_hash[:12]}"
    version_dir = artifacts_dir / version_id
    manifest_path = version_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_hash") != dataset_hash:
            raise ValueError("Le manifeste existant ne correspond pas à son identifiant de version.")
        _write_json(artifacts_dir / "latest.json", {"version_id": version_id})
        return manifest

    manifest = {
        "schema_version": 1,
        "version_id": version_id,
        "dataset_hash": dataset_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "mvtec-ad/bottle",
        "dataset_root": str(dataset_dir),
        "config": asdict(config),
        "preprocessing": {
            "color_mode": "RGB",
            "target_size": [config.target_size, config.target_size],
            "resize_strategy": "letterbox_padding",
            "interpolation": config.interpolation,
            "pixel_scaling": "[0, 1]",
            "padding_value": config.padding_value,
            "statistics_scope": "train_only",
        },
        "normalization": normalization,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "leakage_check": leakage,
        "augmentation_policy": {
            "scope": "train_only",
            "horizontal_flip_probability": 0.5,
            "vertical_flip": False,
            "rotation_degrees": 5,
            "translation_fraction": 0.03,
            "brightness_range": [0.9, 1.1],
            "contrast_range": [0.9, 1.1],
            "justification": (
                "Transformations faibles compatibles avec une bouteille centrée. "
                "Le retournement vertical est interdit, car il est incohérent avec l'orientation physique."
            ),
        },
        "label_policy": {
            "train": "normal uniquement",
            "validation": "images normales réservées depuis train/good et défauts réservés par classe depuis test",
            "test": "images normales et défauts finaux, jamais utilisés pour l'apprentissage ni le réglage",
            "annotation_levels": ["image", "pixel_mask"],
            "ambiguous_cases": "à documenter et à faire arbitrer par un expert métier",
        },
        "evaluation_policy": {
            "image_metrics": ["recall", "f1", "average_precision", "auroc", "confusion_matrix"],
            "pixel_metrics": ["pixel_auroc", "pixel_average_precision"],
            "forbidden_primary_metric": "accuracy",
            "early_stopping": "sous-ensemble normal de validation uniquement",
            "threshold_calibration": "images normales de validation uniquement, sans consulter les défauts ni le test final",
        },
        "limitations": [
            "Le jeu de données ne fournit pas d'identifiant de pièce ou de série : le contrôle de fuite repose sur SHA-256.",
            "Les augmentations doivent être appliquées à la volée et uniquement aux enregistrements du jeu d'entraînement.",
        ],
        "images": records,
    }

    version_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    _write_json(
        version_dir / "splits.json",
        {split: [record["path"] for record in records if record["split"] == split] for split in ("train", "validation", "test")},
    )
    _write_json(version_dir / "normalization.json", normalization)
    _write_json(
        version_dir / "labels.json",
        {
            "classes": class_counts,
            "masks": {record["path"]: record["mask_path"] for record in records if record["mask_path"]},
        },
    )
    _write_json(artifacts_dir / "latest.json", {"version_id": version_id})
    return manifest


def load_latest_preparation(artifacts_dir: Path) -> dict[str, Any] | None:
    """Charge la dernière version préparée en bloquant tout chemin non attendu."""

    latest_path = artifacts_dir / "latest.json"
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    version_id = str(latest.get("version_id", ""))
    if not version_id.startswith("bottle-") or any(char in version_id for char in "/\\."):
        return None
    manifest_path = artifacts_dir / version_id / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def preparation_summary(manifest: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    """Réduit le manifeste aux informations utiles à l'interface."""

    manifest_path = project_dir / "artifacts" / "vision-datasets" / "bottle" / manifest["version_id"] / "manifest.json"
    return {
        "version_id": manifest["version_id"],
        "dataset_hash": manifest["dataset_hash"],
        "created_at": manifest["created_at"],
        "manifest_path": str(manifest_path.relative_to(project_dir)).replace("\\", "/"),
        "target_size": manifest["config"]["target_size"],
        "validation_ratio": manifest["config"]["validation_ratio"],
        "random_seed": manifest["config"]["random_seed"],
        "split_counts": manifest["split_counts"],
        "class_counts": manifest["class_counts"],
        "channel_mean": manifest["normalization"]["channel_mean"],
        "channel_std": manifest["normalization"]["channel_std"],
        "resize_strategy": manifest["preprocessing"]["resize_strategy"],
        "pixel_scaling": manifest["preprocessing"]["pixel_scaling"],
        "leakage_free": not manifest["leakage_check"]["cross_split_duplicates"],
        "augmentation_scope": manifest["augmentation_policy"]["scope"],
        "vertical_flip": manifest["augmentation_policy"]["vertical_flip"],
    }


def preprocess_image(image: Image.Image, config: VisionPreparationConfig) -> Image.Image:
    """Convertit en RGB et redimensionne avec remplissage sans déformer le contenu."""

    config.validate()
    image = ImageOps.exif_transpose(image).convert("RGB")
    resampling = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "nearest": Image.Resampling.NEAREST,
    }[config.interpolation]
    contained = ImageOps.contain(image, (config.target_size, config.target_size), method=resampling)
    canvas = Image.new("RGB", (config.target_size, config.target_size), color=(config.padding_value,) * 3)
    offset = ((config.target_size - contained.width) // 2, (config.target_size - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas


def load_model_input(
    dataset_dir: Path,
    record: dict[str, Any],
    config: VisionPreparationConfig,
    *,
    augment: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Charge un tenseur HWC float32 dans [0, 1], prêt pour un modèle d'images."""

    if augment and (record.get("split") != "train" or record.get("is_anomaly")):
        raise ValueError("L'augmentation est réservée aux images normales du jeu d'entraînement.")
    image_path = (dataset_dir / str(record["path"])).resolve()
    if not image_path.is_relative_to(dataset_dir.resolve()):
        raise ValueError("Chemin image hors du dataset.")
    with Image.open(image_path) as source:
        prepared = preprocess_image(source, config)
    if augment:
        prepared = augment_training_image(prepared, seed)
    return np.asarray(prepared, dtype=np.float32) / np.float32(255.0)


def augment_training_image(image: Image.Image, seed: int) -> Image.Image:
    """Applique uniquement les transformations faibles autorisées aux images saines d'entraînement."""

    generator = random.Random(seed)
    if generator.random() < 0.5:
        image = ImageOps.mirror(image)
    angle = generator.uniform(-5.0, 5.0)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    max_shift = round(image.width * 0.03)
    shift_x = generator.randint(-max_shift, max_shift)
    shift_y = generator.randint(-max_shift, max_shift)
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.9, 1.1))
    return ImageEnhance.Contrast(image).enhance(generator.uniform(0.9, 1.1))


def _validate_mvtec_structure(dataset_dir: Path) -> None:
    required = [dataset_dir / "train" / "good", dataset_dir / "test" / "good", dataset_dir / "ground_truth"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise ValueError(f"Structure MVTec incomplète : {', '.join(missing)}")
    if not any((dataset_dir / "train" / "good").glob("*.png")):
        raise ValueError("Le dossier train/good ne contient aucune image PNG.")


def _stable_split_key(relative_path: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()


def _matching_mask(dataset_dir: Path, label: str, stem: str) -> Path | None:
    candidate = dataset_dir / "ground_truth" / label / f"{stem}_mask.png"
    return candidate if candidate.exists() else None


def _image_record(
    dataset_dir: Path,
    path: Path,
    *,
    split: str,
    label: str,
    is_anomaly: bool,
    mask_path: Path | None = None,
) -> dict[str, Any]:
    relative_path = path.relative_to(dataset_dir).as_posix()
    relative_mask = mask_path.relative_to(dataset_dir).as_posix() if mask_path else None
    return {
        "path": relative_path,
        "split": split,
        "label": label,
        "is_anomaly": is_anomaly,
        "sha256": _file_sha256(path),
        "mask_path": relative_mask,
        "mask_sha256": _file_sha256(mask_path) if mask_path else None,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leakage_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_hash.setdefault(record["sha256"], []).append(record)
    duplicates = []
    for sha256, matches in by_hash.items():
        splits = sorted({match["split"] for match in matches})
        if len(splits) > 1:
            duplicates.append({"sha256": sha256, "splits": splits, "paths": [match["path"] for match in matches]})
    return {
        "method": "SHA-256 du contenu",
        "checked_images": len(records),
        "cross_split_duplicates": duplicates,
        "series_metadata_available": False,
    }


def _normalization_statistics(
    dataset_dir: Path,
    records: list[dict[str, Any]],
    config: VisionPreparationConfig,
) -> dict[str, Any]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for record in records:
        values = load_model_input(dataset_dir, record, config)
        flattened = values.reshape(-1, 3)
        channel_sum += flattened.sum(axis=0, dtype=np.float64)
        channel_squared_sum += np.square(flattened, dtype=np.float64).sum(axis=0)
        pixel_count += flattened.shape[0]
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_squared_sum / pixel_count - np.square(mean), 0.0)
    return {
        "channel_order": ["R", "G", "B"],
        "channel_mean": [round(float(value), 8) for value in mean],
        "channel_std": [round(float(value), 8) for value in np.sqrt(variance)],
        "pixel_count": pixel_count,
        "image_count": len(records),
        "computed_on": "train_only_after_letterbox_and_[0,1]_scaling",
    }


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
