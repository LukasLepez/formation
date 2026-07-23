"""Pipeline B7 rejouable : MVTec → auto-encodeur → évaluation → artefacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from indusense import config
from indusense.vision.train import VisionTrainingConfig, train_vision_autoencoder
from indusense.vision.dataset import VisionPreparationConfig, prepare_vision_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne et évalue l'auto-encodeur MVTec de façon reproductible.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("mse", "ssim"), default="mse")
    parser.add_argument("--latent-filters", type=int, default=16)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--dataset-dir", type=Path, default=config.MVTEC_BOTTLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.VISION_RUNS_DIR)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    manifest = prepare_vision_dataset(
        dataset_dir, config.VISION_DATASETS_DIR,
        VisionPreparationConfig(target_size=args.img_size, random_seed=args.seed),
    )
    run_id = f"vision_ae_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    run_dir = args.output_dir.resolve() / run_id
    tracking_uri = "" if args.no_mlflow else f"sqlite:///{(args.output_dir.resolve() / 'mlflow.db').as_posix()}"
    report = train_vision_autoencoder(
        dataset_dir=dataset_dir, manifest=manifest, run_dir=run_dir, run_id=run_id,
        config=VisionTrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            loss_name=args.loss, latent_filters=args.latent_filters,
            threshold_percentile=args.threshold_percentile, random_seed=args.seed,
        ),
        mlflow_tracking_uri=tracking_uri,
    )
    print(json.dumps({"run_id": run_id, "report": str(run_dir / "report.json"), "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
