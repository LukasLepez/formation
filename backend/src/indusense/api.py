"""Backend FastAPI pour executer les pipelines InduSense."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from indusense.artifacts import RUN_INDEX_JSON
from indusense.maintenance_ml import DEFAULT_LABEL, MaintenanceMlConfig, hash_dataframe, train_maintenance_models
from indusense.processing.ingestion import (
    DEFAULT_DATABASE_URL,
    GoldDatasetConfig,
    ensure_postgres_stack_running,
    run_layer_pipeline,
)
from indusense.reporting.graphs import generate_graph_report
from indusense.vision.train import VisionTrainingConfig, train_vision_autoencoder
from indusense.vision.patchcore import PatchCoreConfig, train_patchcore
from indusense.vision_dataset import (
    VisionPreparationConfig,
    augment_training_image,
    load_latest_preparation,
    preparation_summary,
    prepare_vision_dataset,
    preprocess_image,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_DIR / "artifacts" / "pipeline-runs"
GOLD_DIR = PROJECT_DIR / "artifacts" / "gold-datasets"
ML_RUNS_DIR = PROJECT_DIR / "artifacts" / "maintenance-ml-runs"
MVTEC_DIR = PROJECT_DIR / "data" / "mvtec"
VISION_ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "vision-datasets" / "bottle"
VISION_MODEL_RUNS_DIR = PROJECT_DIR / "artifacts" / "vision-model-runs"
LOGGER = logging.getLogger(__name__)
DOCKER_START_LOCK = threading.Lock()
MLFLOW_UI_PROCESS: subprocess.Popen[str] | None = None
MLFLOW_UI_URL = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "maintenance_predictive_b5"
MLFLOW_REGISTERED_MODEL_NAME = "InduSense_PanneDetection"
MLFLOW_UI_LOG_PATH = ML_RUNS_DIR / "mlflow-ui.log"

RunStatus = Literal["queued", "running", "success", "failed"]
LayerName = Literal["all", "bronze", "silver", "gold"]
GraphSourceLayer = Literal["bronze", "silver"]
TABLES_BY_LAYER = {
    "bronze": ["telemetry_raw", "incidents_raw", "machine", "maintenance"],
    "silver": ["telemetry", "incidents", "machine", "maintenance"],
    "gold": ["gold_dataset"],
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(PROJECT_DIR / ".env")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ML_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    VISION_MODEL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if os.getenv("INDUSENSE_API_START_DOCKER", "1") != "0":
        threading.Thread(target=start_docker_compose_for_api, daemon=True).start()
    yield


app = FastAPI(title="InduSense Pipeline API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunCreate(BaseModel):
    """Parametres acceptes pour lancer un pipeline."""

    layer: LayerName = "all"
    persist_db: bool = True
    auto_start_docker: bool = True
    log_level: str = "INFO"
    database_url: str | None = None


class RunInfo(BaseModel):
    """Etat persiste d'un run."""

    run_id: str
    status: RunStatus
    layer: LayerName
    persist_db: bool
    auto_start_docker: bool
    log_level: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    rows: int | None = None
    columns: int | None = None
    error: str | None = None
    run_dir: str
    log_path: str


class LogsResponse(BaseModel):
    """Extrait incremental de logs."""

    run_id: str
    offset: int = Field(ge=0)
    next_offset: int = Field(ge=0)
    text: str
    status: RunStatus


class GraphRunCreate(BaseModel):
    """Parametres acceptes pour lancer une generation de graphes."""

    source_layer: GraphSourceLayer = "silver"
    database_url: str | None = None
    log_level: str = "INFO"


class GraphRunInfo(BaseModel):
    """Etat persiste d'une generation de graphes."""

    run_id: str
    status: RunStatus
    source_layer: GraphSourceLayer
    log_level: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    graph_count: int | None = None
    incident_rows: int | None = None
    telemetry_rows: int | None = None
    machines: int | None = None
    report_path: str | None = None
    error: str | None = None
    run_dir: str
    log_path: str


class TablePreview(BaseModel):
    layer: str
    table: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    limit: int
    offset: int


class ArtifactRun(BaseModel):
    run_name: str | None = None
    run_ts: str | None = None
    layer: str | None = None
    source_layer: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    run_dir: str | None = None
    report_path: str | None = None
    graphs_dir: str | None = None
    gold_csv_path: str | None = None
    nombre_lignes: int | None = None
    nombre_lignes_telemetrie_lues: int | None = None
    machines_uniques: int | None = None
    nombre_colonnes: int | None = None
    nombre_graphes: int | None = None

    model_config = {"populate_by_name": True}


class GoldCsvInfo(BaseModel):
    run_name: str
    csv_path: str
    created_at: str | None = None
    rows: int | None = None
    columns: int | None = None
    size_bytes: int


class VisionDatasetSplit(BaseModel):
    name: str
    label: str
    count: int
    sample_paths: list[str] = Field(default_factory=list)


class VisionDatasetPreparation(BaseModel):
    version_id: str
    dataset_hash: str
    created_at: str
    manifest_path: str
    target_size: int
    validation_ratio: float
    random_seed: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    channel_mean: list[float]
    channel_std: list[float]
    resize_strategy: str
    pixel_scaling: str
    leakage_free: bool
    augmentation_scope: str
    vertical_flip: bool


class VisionDatasetPrepareRequest(BaseModel):
    target_size: int = Field(default=256, ge=64, le=1024)
    validation_ratio: float = Field(default=0.2, ge=0.05, le=0.4)
    defect_validation_ratio: float = Field(default=0.3, ge=0.1, le=0.5)
    random_seed: int = 42
    padding_value: int = Field(default=0, ge=0, le=255)
    interpolation: Literal["bilinear", "bicubic", "nearest"] = "bilinear"


class VisionDatasetInfo(BaseModel):
    name: str
    root_path: str
    archive_path: str | None = None
    archive_size_bytes: int | None = None
    image_size: str
    total_images: int
    train_good: VisionDatasetSplit
    test_good: VisionDatasetSplit
    test_defects: list[VisionDatasetSplit]
    ground_truth_masks: list[VisionDatasetSplit]
    validation_hint: str
    preparation: VisionDatasetPreparation | None = None


class VisionModelRunCreate(BaseModel):
    model_type: Literal["autoencoder", "patchcore"] = "autoencoder"
    epochs: int = Field(default=20, ge=1, le=200)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=0.1)
    loss_name: Literal["mse", "ssim"] = "mse"
    latent_filters: int = Field(default=16, ge=1, le=128)
    threshold_percentile: float = Field(default=99.0, ge=50, le=100)
    early_stopping_patience: int = Field(default=5, ge=1, le=30)
    random_seed: int = 42
    patchcore_coreset_ratio: float = Field(default=0.05, ge=0.001, le=1.0)
    patchcore_max_memory_patches: int = Field(default=1024, ge=32, le=20_000)
    patchcore_candidate_patches: int = Field(default=10_000, ge=512, le=100_000)


class VisionModelRunInfo(BaseModel):
    run_id: str
    status: RunStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    dataset_version: str
    model_type: Literal["autoencoder", "patchcore"] = "autoencoder"
    epochs: int
    batch_size: int
    loss_name: str
    threshold_percentile: float
    error: str | None = None
    run_dir: str
    log_path: str
    report_path: str | None = None


class MaintenanceMlRunCreate(BaseModel):
    """Paramètres d'un entraînement B5 depuis un CSV Gold généré."""

    label_column: str = DEFAULT_LABEL
    gold_run_name: str | None = None
    random_forest_balanced: bool = True
    selected_models: list[str] = Field(default_factory=lambda: ["logistic_regression", "decision_tree", "random_forest", "random_forest_balanced", "xgboost"])
    decision_tree_max_depth: int = 6
    decision_tree_min_samples_leaf: int = 10
    random_forest_n_estimators: int = 60
    random_forest_max_depth: int = 12
    random_forest_min_samples_leaf: int = 2
    random_forest_min_samples_split: int = 10
    random_forest_max_features: str | None = "sqrt"
    random_forest_bootstrap: bool = True
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_scale_pos_weight_auto: bool = True
    xgboost_scale_pos_weight: float | None = None
    threshold_strategy: str = "balanced"
    target_recall: float = 0.8
    false_negative_cost: float = 20.0
    false_positive_cost: float = 1.0
    experiment_hypothesis: str = ""
    random_state: int = 42
    tune: bool = False
    tune_n_trials: int = Field(default=15, ge=1, le=100)
    tune_timeout_seconds: int = Field(default=600, ge=30, le=7200)
    tune_mode: Literal["frugal", "heavy"] = "frugal"


class MaintenanceMlRunInfo(BaseModel):
    """État persisté d'un entraînement de maintenance prédictive."""

    run_id: str
    status: RunStatus
    label_column: str
    gold_run_name: str | None = None
    random_forest_balanced: bool = True
    selected_models: list[str] = Field(default_factory=lambda: ["logistic_regression", "decision_tree", "random_forest", "random_forest_balanced", "xgboost"])
    decision_tree_max_depth: int = 6
    decision_tree_min_samples_leaf: int = 10
    random_forest_n_estimators: int = 60
    random_forest_max_depth: int = 12
    random_forest_min_samples_leaf: int = 2
    random_forest_min_samples_split: int = 10
    random_forest_max_features: str | None = "sqrt"
    random_forest_bootstrap: bool = True
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_scale_pos_weight_auto: bool = True
    xgboost_scale_pos_weight: float | None = None
    threshold_strategy: str = "balanced"
    target_recall: float = 0.8
    false_negative_cost: float = 20.0
    false_positive_cost: float = 1.0
    experiment_hypothesis: str = ""
    random_state: int = 42
    tune: bool = False
    tune_n_trials: int = 15
    tune_timeout_seconds: int = 600
    tune_mode: str = "frugal"
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    rows: int | None = None
    features: int | None = None
    best_model: str | None = None
    error: str | None = None
    run_dir: str
    log_path: str


class MaintenanceMlReport(BaseModel):
    run_id: str
    status: str
    gold_run_name: str
    gold_csv_path: str
    label_column: str
    rows: int
    features: int
    class_balance: dict[str, Any]
    pr_auc_random_baseline: float | None = None
    scale_pos_weight: float
    xgboost_effective_scale_pos_weight: float | None = None
    random_forest_balanced: bool = True
    selected_models: list[str] = Field(default_factory=lambda: ["logistic_regression", "decision_tree", "random_forest", "random_forest_balanced", "xgboost"])
    decision_tree_max_depth: int = 6
    decision_tree_min_samples_leaf: int = 10
    random_forest_n_estimators: int = 60
    random_forest_max_depth: int = 12
    random_forest_min_samples_leaf: int = 2
    random_forest_min_samples_split: int = 10
    random_forest_max_features: str | None = "sqrt"
    random_forest_bootstrap: bool = True
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_scale_pos_weight_auto: bool = True
    xgboost_scale_pos_weight: float | None = None
    threshold_strategy: str = "balanced"
    target_recall: float = 0.8
    false_negative_cost: float = 20.0
    false_positive_cost: float = 1.0
    experiment_hypothesis: str = ""
    random_state: int = 42
    mlflow_tracking_uri: str
    tuning: dict[str, Any] = Field(default_factory=dict)
    carbon: dict[str, Any] = Field(default_factory=dict)
    b7_artifacts: dict[str, str] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    event_log_path: str | None = None
    results: list[dict[str, Any]]
    best_model: str
    conclusion: str


class MaintenanceMlEvent(BaseModel):
    ts: str
    step: str
    status: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class MlflowRunSummary(BaseModel):
    run_id: str
    app_run_id: str | None = None
    run_name: str | None = None
    status: str | None = None
    start_time: str | None = None
    model: str | None = None
    threshold: float | None = None
    threshold_strategy: str | None = None
    dataset_hash: str | None = None
    validation_pr_auc: float | None = None
    validation_recall: float | None = None
    test_pr_auc: float | None = None
    test_recall: float | None = None
    test_f1: float | None = None
    test_business_cost: float | None = None


class MlflowTrackingSummary(BaseModel):
    experiment_name: str
    tracking_uri: str
    runs: list[MlflowRunSummary]


class MlflowUiStatus(BaseModel):
    running: bool
    url: str


class ModelPromotionResponse(BaseModel):
    registered_model_name: str
    version: str
    stage: str
    mlflow_run_id: str
    model_uri: str
    readme_path: str | None = None


class ModelCandidateTest(BaseModel):
    name: str
    passed: bool
    detail: str


class ModelCandidateTestReport(BaseModel):
    registered_model_name: str
    stage: str
    model_uri: str
    passed: bool
    tests: list[ModelCandidateTest]


def main() -> None:
    """Lance l'API locale via ``uv run indusense-api``."""

    import uvicorn

    uvicorn.run("indusense.api:app", host="127.0.0.1", port=8000, reload=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    return TABLES_BY_LAYER


@app.get("/datasets/{layer}/{table}", response_model=TablePreview)
def preview_table(layer: str, table: str, limit: int = 100, offset: int = 0, database_url: str | None = None) -> TablePreview:
    if layer not in TABLES_BY_LAYER:
        raise HTTPException(status_code=404, detail="Couche inconnue.")
    if table not in TABLES_BY_LAYER[layer]:
        raise HTTPException(status_code=404, detail="Table inconnue pour cette couche.")

    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    engine = create_engine(database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL)
    try:
        count_query = text(f'SELECT COUNT(*) FROM "{layer}"."{table}"')
        data_query = text(f'SELECT * FROM "{layer}"."{table}" LIMIT :limit OFFSET :offset')
        with engine.begin() as connection:
            total_rows = int(connection.execute(count_query).scalar() or 0)
            frame = pd.read_sql_query(data_query, connection, params={"limit": safe_limit, "offset": safe_offset})
    except Exception as error:  # noqa: BLE001 - l'API transforme l'erreur SQL en message frontend.
        raise HTTPException(status_code=503, detail=f"Impossible de lire {layer}.{table} : {error}") from error

    return TablePreview(
        layer=layer,
        table=table,
        columns=list(frame.columns),
        rows=json_safe_records(frame),
        total_rows=total_rows,
        limit=safe_limit,
        offset=safe_offset,
    )


@app.get("/artifact-runs", response_model=list[ArtifactRun])
def list_artifact_runs() -> list[ArtifactRun]:
    if not RUN_INDEX_JSON.exists():
        return []
    runs = json.loads(RUN_INDEX_JSON.read_text(encoding="utf-8"))
    return [ArtifactRun(**run) for run in sorted(runs, key=lambda item: item.get("run_ts", ""), reverse=True)]


@app.get("/gold-csvs", response_model=list[GoldCsvInfo])
def list_gold_csvs() -> list[GoldCsvInfo]:
    if not GOLD_DIR.exists():
        return []

    csvs = []
    for csv_path in sorted(GOLD_DIR.glob("*_gold_dataset/gold_dataset_*.csv"), reverse=True):
        run_dir = csv_path.parent
        metadata_path = run_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csvs.append(
            GoldCsvInfo(
                run_name=run_dir.name,
                csv_path=str(csv_path.relative_to(PROJECT_DIR)),
                created_at=metadata.get("run_ts"),
                rows=metadata.get("nombre_lignes"),
                columns=metadata.get("nombre_colonnes"),
                size_bytes=csv_path.stat().st_size,
            )
        )
    return csvs


@app.get("/gold-csvs/{run_name}", response_model=TablePreview)
def preview_gold_csv(run_name: str, limit: int = 100, offset: int = 0) -> TablePreview:
    csv_path = resolve_gold_csv_path(run_name)
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    try:
        total_rows = max(0, sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1)
        frame = pd.read_csv(csv_path, skiprows=range(1, safe_offset + 1), nrows=safe_limit)
    except Exception as error:  # noqa: BLE001 - retourne un message exploitable au frontend.
        raise HTTPException(status_code=503, detail=f"Impossible de lire le CSV Gold {run_name} : {error}") from error

    return TablePreview(
        layer="gold_csv",
        table=run_name,
        columns=list(frame.columns),
        rows=json_safe_records(frame),
        total_rows=total_rows,
        limit=safe_limit,
        offset=safe_offset,
    )


@app.get("/gold-csvs/{run_name}/download")
def download_gold_csv(run_name: str) -> FileResponse:
    csv_path = resolve_gold_csv_path(run_name)
    return FileResponse(
        path=csv_path,
        filename=csv_path.name,
        media_type="text/csv",
    )


@app.get("/vision-datasets/bottle", response_model=VisionDatasetInfo)
def get_bottle_vision_dataset() -> VisionDatasetInfo:
    dataset_dir = MVTEC_DIR / "bottle"
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset MVTec bottle introuvable dans backend/data/mvtec.")

    train_good = vision_split(dataset_dir, "train/good", "Train sain")
    test_good = vision_split(dataset_dir, "test/good", "Test sain")
    test_defects = [
        vision_split(dataset_dir, f"test/{defect_dir.name}", defect_dir.name)
        for defect_dir in sorted((dataset_dir / "test").glob("*"))
        if defect_dir.is_dir() and defect_dir.name != "good"
    ]
    ground_truth_masks = [
        vision_split(dataset_dir, f"ground_truth/{mask_dir.name}", mask_dir.name)
        for mask_dir in sorted((dataset_dir / "ground_truth").glob("*"))
        if mask_dir.is_dir()
    ]
    archive_path = MVTEC_DIR / "bottle.tar.xz"
    total_images = train_good.count + test_good.count + sum(split.count for split in test_defects)
    latest_preparation = load_latest_preparation(VISION_ARTIFACTS_DIR)
    return VisionDatasetInfo(
        name="bottle",
        root_path=str(dataset_dir.relative_to(PROJECT_DIR)),
        archive_path=str(archive_path.relative_to(PROJECT_DIR)) if archive_path.exists() else None,
        archive_size_bytes=archive_path.stat().st_size if archive_path.exists() else None,
        image_size="256 x 256 cible TP",
        total_images=total_images,
        train_good=train_good,
        test_good=test_good,
        test_defects=test_defects,
        ground_truth_masks=ground_truth_masks,
        validation_hint="Réserver 15 à 20 % des images saines d'entraînement pour calibrer le seuil en partie 2.",
        preparation=preparation_summary(latest_preparation, PROJECT_DIR) if latest_preparation else None,
    )


@app.get("/vision-datasets/bottle/image")
def get_bottle_vision_image(path: str) -> FileResponse:
    image_path = resolve_vision_image_path(path)
    return FileResponse(path=image_path, media_type="image/png")


@app.post("/vision-datasets/bottle/prepare", response_model=VisionDatasetPreparation)
def prepare_bottle_vision_dataset(payload: VisionDatasetPrepareRequest) -> VisionDatasetPreparation:
    dataset_dir = MVTEC_DIR / "bottle"
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset MVTec bottle introuvable dans backend/data/mvtec.")
    try:
        manifest = prepare_vision_dataset(
            dataset_dir,
            VISION_ARTIFACTS_DIR,
            VisionPreparationConfig(**payload.model_dump()),
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return VisionDatasetPreparation(**preparation_summary(manifest, PROJECT_DIR))


@app.get("/vision-datasets/bottle/prepared-image")
def get_bottle_prepared_image(path: str, augmented: bool = False, seed: int = 42) -> Response:
    """Montre exactement le pretraitement utilise par le futur chargeur Deep Learning."""

    image_path = resolve_vision_image_path(path)
    manifest = load_latest_preparation(VISION_ARTIFACTS_DIR)
    config = VisionPreparationConfig(**manifest["config"]) if manifest else VisionPreparationConfig()
    if augmented:
        if not manifest:
            raise HTTPException(status_code=409, detail="Prépare d'abord le dataset avant de demander une augmentation.")
        record = next((item for item in manifest["images"] if item["path"] == path), None)
        if not record or record["split"] != "train" or record["is_anomaly"]:
            raise HTTPException(status_code=400, detail="L'augmentation est autorisée uniquement sur le split train sain.")

    with Image.open(image_path) as source:
        prepared = preprocess_image(source, config)
    if augmented:
        prepared = augment_training_image(prepared, seed)
    output = BytesIO()
    prepared.save(output, format="PNG")
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store" if augmented else "private, max-age=3600"},
    )


@app.post("/vision-model-runs", response_model=VisionModelRunInfo)
def create_vision_model_run(payload: VisionModelRunCreate) -> VisionModelRunInfo:
    manifest = load_latest_preparation(VISION_ARTIFACTS_DIR)
    if not manifest:
        raise HTTPException(status_code=409, detail="Préparez d'abord le jeu de données MVTec bottle.")
    model_prefix = "vision_patchcore_" if payload.model_type == "patchcore" else "vision_ae_"
    run_id = model_prefix + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = VISION_MODEL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "vision_training.log"
    log_path.write_text("", encoding="utf-8")
    info = VisionModelRunInfo(
        run_id=run_id,
        status="queued",
        created_at=utc_now(),
        dataset_version=manifest["version_id"],
        model_type=payload.model_type,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        loss_name=payload.loss_name,
        threshold_percentile=payload.threshold_percentile,
        run_dir=str(run_dir.relative_to(PROJECT_DIR)).replace("\\", "/"),
        log_path=str(log_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
    )
    write_vision_model_metadata(run_dir, info)
    threading.Thread(target=execute_vision_model_run, args=(run_dir, payload), daemon=True).start()
    return info


@app.get("/vision-model-runs", response_model=list[VisionModelRunInfo])
def list_vision_model_runs() -> list[VisionModelRunInfo]:
    runs = []
    if not VISION_MODEL_RUNS_DIR.exists():
        return runs
    for metadata_path in sorted(VISION_MODEL_RUNS_DIR.glob("vision_*/metadata.json"), reverse=True):
        try:
            runs.append(VisionModelRunInfo(**json.loads(metadata_path.read_text(encoding="utf-8"))))
        except Exception:
            LOGGER.warning("Métadonnée de vision ignorée, car elle est illisible : %s", metadata_path)
    return runs


@app.get("/vision-model-runs/{run_id}", response_model=VisionModelRunInfo)
def get_vision_model_run(run_id: str) -> VisionModelRunInfo:
    return read_vision_model_metadata(resolve_vision_model_run_dir(run_id))


@app.get("/vision-model-runs/{run_id}/report")
def get_vision_model_report(run_id: str) -> dict[str, Any]:
    run_dir = resolve_vision_model_run_dir(run_id)
    info = read_vision_model_metadata(run_dir)
    if info.status != "success":
        raise HTTPException(status_code=409, detail="Le rapport sera disponible à la fin du run.")
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Rapport de vision introuvable.")
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.get("/vision-model-runs/{run_id}/logs", response_class=PlainTextResponse)
def get_vision_model_logs(run_id: str) -> str:
    run_dir = resolve_vision_model_run_dir(run_id)
    log_path = run_dir / "vision_training.log"
    return log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""


@app.get("/vision-model-runs/{run_id}/artifacts/{artifact_path:path}")
def get_vision_model_artifact(run_id: str, artifact_path: str) -> FileResponse:
    run_dir = resolve_vision_model_run_dir(run_id)
    if "\\" in artifact_path or ".." in Path(artifact_path).parts:
        raise HTTPException(status_code=400, detail="Chemin d'artefact non valide.")
    resolved = (run_dir / artifact_path).resolve()
    if not resolved.is_relative_to(run_dir.resolve()) or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artefact de vision introuvable.")
    media_type = "image/png" if resolved.suffix.lower() == ".png" else "application/octet-stream"
    return FileResponse(resolved, media_type=media_type, filename=resolved.name)


@app.post("/maintenance-ml-runs", response_model=MaintenanceMlRunInfo)
def create_maintenance_ml_run(payload: MaintenanceMlRunCreate) -> MaintenanceMlRunInfo:
    run_id = "maintenance_ml_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = ML_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "maintenance_ml.log"
    log_path.write_text("", encoding="utf-8")

    info = MaintenanceMlRunInfo(
        run_id=run_id,
        status="queued",
        label_column=payload.label_column,
        gold_run_name=payload.gold_run_name,
        random_forest_balanced=payload.random_forest_balanced,
        selected_models=payload.selected_models,
        decision_tree_max_depth=payload.decision_tree_max_depth,
        decision_tree_min_samples_leaf=payload.decision_tree_min_samples_leaf,
        random_forest_n_estimators=payload.random_forest_n_estimators,
        random_forest_max_depth=payload.random_forest_max_depth,
        random_forest_min_samples_leaf=payload.random_forest_min_samples_leaf,
        random_forest_min_samples_split=payload.random_forest_min_samples_split,
        random_forest_max_features=payload.random_forest_max_features,
        random_forest_bootstrap=payload.random_forest_bootstrap,
        xgboost_n_estimators=payload.xgboost_n_estimators,
        xgboost_max_depth=payload.xgboost_max_depth,
        xgboost_learning_rate=payload.xgboost_learning_rate,
        xgboost_scale_pos_weight_auto=payload.xgboost_scale_pos_weight_auto,
        xgboost_scale_pos_weight=payload.xgboost_scale_pos_weight,
        threshold_strategy=payload.threshold_strategy,
        target_recall=payload.target_recall,
        false_negative_cost=payload.false_negative_cost,
        false_positive_cost=payload.false_positive_cost,
        experiment_hypothesis=payload.experiment_hypothesis,
        random_state=payload.random_state,
        tune=payload.tune,
        tune_n_trials=payload.tune_n_trials,
        tune_timeout_seconds=payload.tune_timeout_seconds,
        tune_mode=payload.tune_mode,
        created_at=utc_now(),
        run_dir=str(run_dir.relative_to(PROJECT_DIR)),
        log_path=str(log_path.relative_to(PROJECT_DIR)),
    )
    write_ml_metadata(run_dir, info)
    threading.Thread(target=execute_maintenance_ml_run, args=(run_dir, payload), daemon=True).start()
    return info


@app.get("/maintenance-ml-runs", response_model=list[MaintenanceMlRunInfo])
def list_maintenance_ml_runs() -> list[MaintenanceMlRunInfo]:
    runs = []
    for path in sorted(ML_RUNS_DIR.glob("maintenance_ml_*/metadata.json"), reverse=True):
        try:
            runs.append(MaintenanceMlRunInfo(**json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            LOGGER.warning("Le fichier de métadonnées ML %s a été ignoré car il est illisible.", path)
    return runs


@app.get("/maintenance-ml-runs/{run_id}", response_model=MaintenanceMlRunInfo)
def get_maintenance_ml_run(run_id: str) -> MaintenanceMlRunInfo:
    return read_existing_ml_metadata(run_id)


@app.delete("/maintenance-ml-runs/{run_id}", response_model=dict[str, str])
def delete_maintenance_ml_run(run_id: str) -> dict[str, str]:
    info = read_existing_ml_metadata(run_id)
    if info.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Impossible de supprimer un run ML encore en cours.")
    run_dir = PROJECT_DIR / info.run_dir
    if not run_dir.resolve().is_relative_to(ML_RUNS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Chemin du run ML invalide.")
    shutil.rmtree(run_dir)
    return {"status": "deleted", "run_id": run_id}


@app.get("/maintenance-ml-runs/{run_id}/report", response_model=MaintenanceMlReport)
def get_maintenance_ml_report(run_id: str) -> MaintenanceMlReport:
    info = read_existing_ml_metadata(run_id)
    report_path = PROJECT_DIR / info.run_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Rapport ML introuvable pour ce run.")
    return MaintenanceMlReport(**json.loads(report_path.read_text(encoding="utf-8")))


@app.get("/maintenance-ml-runs/{run_id}/logs/raw", response_class=PlainTextResponse)
def get_raw_maintenance_ml_logs(run_id: str) -> str:
    info = read_existing_ml_metadata(run_id)
    log_path = PROJECT_DIR / info.log_path
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Fichier de logs ML introuvable.")
    return log_path.read_text(encoding="utf-8", errors="replace")


@app.get("/maintenance-ml-runs/{run_id}/events", response_model=list[MaintenanceMlEvent])
def get_maintenance_ml_events(run_id: str) -> list[MaintenanceMlEvent]:
    info = read_existing_ml_metadata(run_id)
    events_path = PROJECT_DIR / info.run_dir / "training_events.jsonl"
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(MaintenanceMlEvent(**json.loads(line)))
        except Exception:
            LOGGER.warning("Une ligne du journal structuré ML %s est illisible.", events_path)
    return events


@app.get("/maintenance-ml-runs/{run_id}/artifacts/{artifact_path:path}")
def get_maintenance_ml_artifact(run_id: str, artifact_path: str) -> FileResponse:
    info = read_existing_ml_metadata(run_id)
    run_dir = (PROJECT_DIR / info.run_dir).resolve()
    path = (run_dir / artifact_path).resolve()
    if not path.is_relative_to(run_dir) or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artefact ML introuvable.")
    return FileResponse(path=path)


@app.get("/maintenance-mlflow/runs", response_model=MlflowTrackingSummary)
def list_maintenance_mlflow_runs(limit: int = 50) -> MlflowTrackingSummary:
    experiment_name = MLFLOW_EXPERIMENT_NAME
    tracking_uri = f"sqlite:///{(ML_RUNS_DIR / 'mlflow' / 'mlflow.db').as_posix()}"
    report_fallbacks = mlflow_report_fallbacks()
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        frame = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.test_business_cost ASC", "metrics.validation_pr_auc DESC"],
            max_results=max(1, min(limit, 200)),
        )
    except Exception as error:  # noqa: BLE001 - l'UI explique l'absence de tracking au lieu de casser.
        LOGGER.warning("Lecture MLflow impossible : %s", error)
        return MlflowTrackingSummary(experiment_name=experiment_name, tracking_uri=tracking_uri, runs=[])

    runs = []
    for record in frame.to_dict(orient="records"):
        gold_csv = optional_str(record.get("params.gold_csv"))
        model = optional_str(record.get("params.model"))
        threshold = optional_float(record.get("params.threshold"))
        fallback = report_fallbacks.get(mlflow_fallback_key(gold_csv, model, threshold), {})
        runs.append(
            MlflowRunSummary(
                run_id=str(record.get("run_id", "")),
                app_run_id=optional_str(record.get("tags.app_run_id")) or optional_str(fallback.get("app_run_id")),
                run_name=optional_str(record.get("tags.mlflow.runName")),
                status=optional_str(record.get("status")),
                start_time=optional_iso(record.get("start_time")),
                model=model,
                threshold=threshold,
                threshold_strategy=optional_str(record.get("params.threshold_strategy")) or optional_str(fallback.get("threshold_strategy")),
                dataset_hash=optional_str(record.get("tags.dataset_hash")) or optional_str(fallback.get("dataset_hash")),
                validation_pr_auc=optional_float(record.get("metrics.validation_pr_auc")),
                validation_recall=optional_float(record.get("metrics.validation_recall")),
                test_pr_auc=optional_float(record.get("metrics.test_pr_auc")),
                test_recall=optional_float(record.get("metrics.test_recall")),
                test_f1=optional_float(record.get("metrics.test_f1")),
                test_business_cost=optional_float(record.get("metrics.test_business_cost")) or optional_float(fallback.get("test_business_cost")),
            )
        )
    return MlflowTrackingSummary(experiment_name=experiment_name, tracking_uri=tracking_uri, runs=runs)


@app.get("/maintenance-mlflow/ui", response_model=MlflowUiStatus)
def get_mlflow_ui_status() -> MlflowUiStatus:
    return MlflowUiStatus(running=port_is_open("127.0.0.1", 5000), url=MLFLOW_UI_URL)


@app.post("/maintenance-mlflow/ui/start", response_model=MlflowUiStatus)
def start_mlflow_ui() -> MlflowUiStatus:
    global MLFLOW_UI_PROCESS  # noqa: PLW0603 - processus local contrôlé par l'API.
    if not port_is_open("127.0.0.1", 5000):
        tracking_uri = f"sqlite:///{(ML_RUNS_DIR / 'mlflow' / 'mlflow.db').as_posix()}"
        MLFLOW_UI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_uri,
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        log_file = MLFLOW_UI_LOG_PATH.open("a", encoding="utf-8")
        MLFLOW_UI_PROCESS = subprocess.Popen(  # noqa: S603 - commande construite sans entrée utilisateur.
            command,
            cwd=str(PROJECT_DIR),
            stdout=log_file,
            stderr=log_file,
            text=True,
            env=env,
        )
        for _ in range(30):
            if port_is_open("127.0.0.1", 5000):
                return MlflowUiStatus(running=True, url=MLFLOW_UI_URL)
            if MLFLOW_UI_PROCESS.poll() is not None:
                raise HTTPException(status_code=503, detail=f"MLflow UI n'a pas démarré : {tail_text(MLFLOW_UI_LOG_PATH)}")
            time.sleep(0.25)
    running = port_is_open("127.0.0.1", 5000)
    if not running:
        raise HTTPException(status_code=503, detail=f"MLflow UI ne répond pas encore : {tail_text(MLFLOW_UI_LOG_PATH)}")
    return MlflowUiStatus(running=running, url=MLFLOW_UI_URL)


@app.post("/maintenance-ml-runs/{run_id}/promote-staging", response_model=ModelPromotionResponse)
def promote_best_model_to_staging(run_id: str) -> ModelPromotionResponse:
    report = get_maintenance_ml_report(run_id).model_dump()
    best = next((result for result in report["results"] if result.get("model") == report["best_model"]), report["results"][0])
    mlflow_run_id = best.get("mlflow_run_id")
    if not mlflow_run_id:
        raise HTTPException(status_code=409, detail="Ce rapport ne contient pas de run_id MLflow pour le meilleur modèle. Relance un entraînement avec le backend à jour.")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = report["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        model_uri = f"runs:/{mlflow_run_id}/model"
        version = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
        client.set_model_version_tag(MLFLOW_REGISTERED_MODEL_NAME, version.version, "app_run_id", run_id)
        client.set_model_version_tag(MLFLOW_REGISTERED_MODEL_NAME, version.version, "dataset_hash", (report.get("reproducibility") or {}).get("dataset_hash", ""))
        client.transition_model_version_stage(
            name=MLFLOW_REGISTERED_MODEL_NAME,
            version=version.version,
            stage="Staging",
            archive_existing_versions=True,
        )
    except Exception as error:  # noqa: BLE001 - renvoyé au front.
        raise HTTPException(status_code=503, detail=f"Promotion MLflow impossible : {error}") from error

    readme_path = write_candidate_readme(run_id, report, best, str(version.version), "Staging")
    return ModelPromotionResponse(
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        version=str(version.version),
        stage="Staging",
        mlflow_run_id=mlflow_run_id,
        model_uri=f"models:/{MLFLOW_REGISTERED_MODEL_NAME}/Staging",
        readme_path=str(readme_path.relative_to(PROJECT_DIR)) if readme_path else None,
    )


@app.post("/maintenance-mlflow/model-candidate/test", response_model=ModelCandidateTestReport)
def test_staging_model_candidate() -> ModelCandidateTestReport:
    model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}/Staging"
    tests: list[ModelCandidateTest] = []
    try:
        import mlflow

        tracking_uri = f"sqlite:///{(ML_RUNS_DIR / 'mlflow' / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        tests.append(ModelCandidateTest(name="Chargement modèle", passed=True, detail=model_uri))
    except Exception as error:  # noqa: BLE001
        tests.append(ModelCandidateTest(name="Chargement modèle", passed=False, detail=str(error)))
        return ModelCandidateTestReport(registered_model_name=MLFLOW_REGISTERED_MODEL_NAME, stage="Staging", model_uri=model_uri, passed=False, tests=tests)

    reports = sorted(ML_RUNS_DIR.glob("maintenance_ml_*/report.json"), reverse=True)
    latest_report = load_latest_success_report_for_registered_model()
    if not latest_report:
        tests.append(ModelCandidateTest(name="Rapport associé", passed=False, detail="Aucun rapport local compatible trouvé."))
        return ModelCandidateTestReport(registered_model_name=MLFLOW_REGISTERED_MODEL_NAME, stage="Staging", model_uri=model_uri, passed=False, tests=tests)

    try:
        csv_path = Path(latest_report["gold_csv_path"])
        gold = pd.read_csv(csv_path).sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)
        label_column = latest_report["label_column"]
        feature_columns = latest_report.get("feature_columns") or [column for column in gold.columns if column != label_column]
        test_frame = gold.loc[gold["split_set"] == "test", feature_columns].head(50)
        single_pred = model.predict(test_frame.head(1))
        batch_pred = model.predict(test_frame)
        tests.append(ModelCandidateTest(name="Prédiction single", passed=len(single_pred) == 1, detail=f"{len(single_pred)} prédiction"))
        tests.append(ModelCandidateTest(name="Prédiction batch", passed=len(batch_pred) == len(test_frame), detail=f"{len(batch_pred)} / {len(test_frame)} prédictions"))
        best = next((result for result in latest_report["results"] if result.get("model") == latest_report["best_model"]), latest_report["results"][0])
        recall = float(best.get("recall_test") or 0)
        target = float(latest_report.get("target_recall") or 0.8)
        tests.append(ModelCandidateTest(name="Recall cible", passed=recall >= min(target, 0.7), detail=f"recall={recall:.3f}, cible={target:.3f}"))
    except Exception as error:  # noqa: BLE001
        tests.append(ModelCandidateTest(name="Prédictions", passed=False, detail=str(error)))

    return ModelCandidateTestReport(
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        stage="Staging",
        model_uri=model_uri,
        passed=all(test.passed for test in tests),
        tests=tests,
    )


@app.get("/maintenance-ml-runs/{run_id}/candidate-readme", response_class=PlainTextResponse)
def get_candidate_readme(run_id: str) -> str:
    info = read_existing_ml_metadata(run_id)
    readme_path = PROJECT_DIR / info.run_dir / "model_candidate_README.md"
    if not readme_path.exists():
        raise HTTPException(status_code=404, detail="Fiche modèle candidate introuvable.")
    return readme_path.read_text(encoding="utf-8")


@app.post("/docker/start")
def start_docker() -> dict[str, str]:
    threading.Thread(target=start_docker_compose_for_api, daemon=True).start()
    return {"status": "starting"}


@app.post("/runs", response_model=RunInfo)
def create_run(payload: RunCreate) -> RunInfo:
    if not payload.persist_db and payload.layer != "gold":
        raise HTTPException(status_code=400, detail="persist_db=false est compatible uniquement avec layer=gold.")

    run_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "pipeline.log"
    log_path.write_text("", encoding="utf-8")

    info = RunInfo(
        run_id=run_id,
        status="queued",
        layer=payload.layer,
        persist_db=payload.persist_db,
        auto_start_docker=payload.auto_start_docker,
        log_level=payload.log_level.upper(),
        created_at=utc_now(),
        run_dir=str(run_dir.relative_to(PROJECT_DIR)),
        log_path=str(log_path.relative_to(PROJECT_DIR)),
    )
    write_metadata(run_dir, info)

    thread = threading.Thread(target=execute_run, args=(run_dir, payload), daemon=True)
    thread.start()
    return info


@app.get("/runs", response_model=list[RunInfo])
def list_runs() -> list[RunInfo]:
    return [read_metadata(path.parent) for path in sorted(RUNS_DIR.glob("*/metadata.json"), reverse=True)]


@app.get("/runs/{run_id}", response_model=RunInfo)
def get_run(run_id: str) -> RunInfo:
    return read_existing_metadata(run_id)


@app.get("/runs/{run_id}/logs", response_model=LogsResponse)
def get_logs(run_id: str, offset: int = 0) -> LogsResponse:
    info = read_existing_metadata(run_id)
    log_path = PROJECT_DIR / info.log_path
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Fichier de logs introuvable.")

    content = log_path.read_text(encoding="utf-8", errors="replace")
    safe_offset = max(0, min(offset, len(content)))
    return LogsResponse(
        run_id=run_id,
        offset=safe_offset,
        next_offset=len(content),
        text=content[safe_offset:],
        status=info.status,
    )


@app.get("/runs/{run_id}/logs/raw", response_class=PlainTextResponse)
def get_raw_logs(run_id: str) -> str:
    info = read_existing_metadata(run_id)
    log_path = PROJECT_DIR / info.log_path
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Fichier de logs introuvable.")
    return log_path.read_text(encoding="utf-8", errors="replace")


@app.post("/graph-runs", response_model=GraphRunInfo)
def create_graph_run(payload: GraphRunCreate) -> GraphRunInfo:
    run_id = "graphs_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "graphs.log"
    log_path.write_text("", encoding="utf-8")

    info = GraphRunInfo(
        run_id=run_id,
        status="queued",
        source_layer=payload.source_layer,
        log_level=payload.log_level.upper(),
        created_at=utc_now(),
        run_dir=str(run_dir.relative_to(PROJECT_DIR)),
        log_path=str(log_path.relative_to(PROJECT_DIR)),
    )
    write_metadata(run_dir, info)
    threading.Thread(target=execute_graph_run, args=(run_dir, payload), daemon=True).start()
    return info


@app.get("/graph-runs", response_model=list[GraphRunInfo])
def list_graph_runs() -> list[GraphRunInfo]:
    runs = []
    for path in sorted(RUNS_DIR.glob("graphs_*/metadata.json"), reverse=True):
        try:
            runs.append(GraphRunInfo(**json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            LOGGER.warning("Le fichier de métadonnées des graphes %s a été ignoré car il est illisible.", path)
    return runs


@app.get("/graph-runs/{run_id}", response_model=GraphRunInfo)
def get_graph_run(run_id: str) -> GraphRunInfo:
    return read_existing_graph_metadata(run_id)


@app.get("/graph-runs/{run_id}/logs", response_model=LogsResponse)
def get_graph_logs(run_id: str, offset: int = 0) -> LogsResponse:
    info = read_existing_graph_metadata(run_id)
    log_path = PROJECT_DIR / info.log_path
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Fichier de logs introuvable.")
    content = log_path.read_text(encoding="utf-8", errors="replace")
    safe_offset = max(0, min(offset, len(content)))
    return LogsResponse(
        run_id=run_id,
        offset=safe_offset,
        next_offset=len(content),
        text=content[safe_offset:],
        status=info.status,
    )


@app.get("/graph-runs/{run_id}/logs/raw", response_class=PlainTextResponse)
def get_raw_graph_logs(run_id: str) -> str:
    info = read_existing_graph_metadata(run_id)
    log_path = PROJECT_DIR / info.log_path
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Fichier de logs introuvable.")
    return log_path.read_text(encoding="utf-8", errors="replace")


def execute_run(run_dir: Path, payload: RunCreate) -> None:
    info = read_metadata(run_dir)
    info.status = "running"
    info.started_at = utc_now()
    write_metadata(run_dir, info)

    log_path = run_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(getattr(logging, payload.log_level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)

    try:
        LOGGER.info(
            "Le run %s démarre. La couche demandée est « %s » et la persistance PostgreSQL est %s.",
            info.run_id,
            payload.layer,
            "activée" if payload.persist_db else "désactivée",
        )
        config = GoldDatasetConfig(
            output_dir=GOLD_DIR,
            persist_db=payload.persist_db,
            database_url=payload.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
            auto_start_docker=payload.auto_start_docker,
            layer=payload.layer,
        )
        result = run_layer_pipeline(config)
        if result is not None:
            info.rows = int(result.shape[0])
            info.columns = int(result.shape[1])
        info.status = "success"
        LOGGER.info("Le run %s s'est terminé avec succès.", info.run_id)
    except Exception as error:  # noqa: BLE001 - on persiste l'erreur du run.
        info.status = "failed"
        info.error = str(error)
        LOGGER.error("Le run %s a échoué : %s", info.run_id, error)
        LOGGER.debug("Traceback du run:\n%s", traceback.format_exc())
    finally:
        info.finished_at = utc_now()
        write_metadata(run_dir, info)
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()


def start_docker_compose_for_api() -> None:
    if not DOCKER_START_LOCK.acquire(blocking=False):
        LOGGER.info("Le démarrage de Docker Compose est déjà en cours.")
        return

    log_path = RUNS_DIR / "backend-docker.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    try:
        LOGGER.info("Le backend démarre Docker Compose pour PostgreSQL et pgAdmin.")
        ensure_postgres_stack_running()
        LOGGER.info("Docker Compose est prêt : PostgreSQL et pgAdmin sont disponibles.")
    except Exception as error:  # noqa: BLE001 - backend disponible meme si Docker echoue.
        LOGGER.error("Docker Compose n'est pas disponible : %s", error)
        LOGGER.debug("Traceback Docker Compose:\n%s", traceback.format_exc())
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()
        DOCKER_START_LOCK.release()


def execute_graph_run(run_dir: Path, payload: GraphRunCreate) -> None:
    info = read_graph_metadata(run_dir)
    info.status = "running"
    info.started_at = utc_now()
    write_metadata(run_dir, info)

    log_path = run_dir / "graphs.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(getattr(logging, payload.log_level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)
    try:
        LOGGER.info("La génération de graphes %s démarre pour la couche « %s ».", info.run_id, payload.source_layer)
        result = generate_graph_report(
            source_layer=payload.source_layer,
            database_url=payload.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
        )
        info.graph_count = result.graph_count
        info.incident_rows = result.incident_rows
        info.telemetry_rows = result.telemetry_rows
        info.machines = result.machines
        info.report_path = str(result.report_path.relative_to(PROJECT_DIR))
        info.status = "success"
        LOGGER.info("La génération de graphes %s est terminée. Le rapport est disponible ici : %s", info.run_id, info.report_path)
    except Exception as error:  # noqa: BLE001 - on persiste l'erreur du run.
        info.status = "failed"
        info.error = str(error)
        LOGGER.error("La génération de graphes %s a échoué : %s", info.run_id, error)
        LOGGER.debug("Traceback graphes:\n%s", traceback.format_exc())
    finally:
        info.finished_at = utc_now()
        write_metadata(run_dir, info)
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()


def execute_maintenance_ml_run(run_dir: Path, payload: MaintenanceMlRunCreate) -> None:
    info = read_ml_metadata(run_dir)
    info.status = "running"
    info.started_at = utc_now()
    write_ml_metadata(run_dir, info)

    log_path = run_dir / "maintenance_ml.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    try:
        LOGGER.info("Le run ML %s démarre avec la cible %s.", info.run_id, payload.label_column)
        report = train_maintenance_models(
            MaintenanceMlConfig(
                gold_dir=GOLD_DIR,
                run_dir=run_dir,
                label_column=payload.label_column,
                gold_run_name=payload.gold_run_name,
                random_forest_balanced=payload.random_forest_balanced,
                selected_models=tuple(payload.selected_models),
                decision_tree_max_depth=payload.decision_tree_max_depth,
                decision_tree_min_samples_leaf=payload.decision_tree_min_samples_leaf,
                random_forest_n_estimators=payload.random_forest_n_estimators,
                random_forest_max_depth=payload.random_forest_max_depth,
                random_forest_min_samples_leaf=payload.random_forest_min_samples_leaf,
                random_forest_min_samples_split=payload.random_forest_min_samples_split,
                random_forest_max_features=payload.random_forest_max_features,
                random_forest_bootstrap=payload.random_forest_bootstrap,
                xgboost_n_estimators=payload.xgboost_n_estimators,
                xgboost_max_depth=payload.xgboost_max_depth,
                xgboost_learning_rate=payload.xgboost_learning_rate,
                xgboost_scale_pos_weight_auto=payload.xgboost_scale_pos_weight_auto,
                xgboost_scale_pos_weight=payload.xgboost_scale_pos_weight,
                threshold_strategy=payload.threshold_strategy,
                target_recall=payload.target_recall,
                false_negative_cost=payload.false_negative_cost,
                false_positive_cost=payload.false_positive_cost,
                experiment_hypothesis=payload.experiment_hypothesis,
                random_state=payload.random_state,
                tune=payload.tune,
                tune_n_trials=payload.tune_n_trials,
                tune_timeout_seconds=payload.tune_timeout_seconds,
                tune_mode=payload.tune_mode,
            )
        )
        info.rows = report["rows"]
        info.features = report["features"]
        info.gold_run_name = report["gold_run_name"]
        info.random_forest_balanced = report.get("random_forest_balanced", payload.random_forest_balanced)
        info.selected_models = report.get("selected_models", payload.selected_models)
        info.decision_tree_max_depth = report.get("decision_tree_max_depth", payload.decision_tree_max_depth)
        info.decision_tree_min_samples_leaf = report.get("decision_tree_min_samples_leaf", payload.decision_tree_min_samples_leaf)
        info.random_forest_n_estimators = report.get("random_forest_n_estimators", payload.random_forest_n_estimators)
        info.random_forest_max_depth = report.get("random_forest_max_depth", payload.random_forest_max_depth)
        info.random_forest_min_samples_leaf = report.get("random_forest_min_samples_leaf", payload.random_forest_min_samples_leaf)
        info.random_forest_min_samples_split = report.get("random_forest_min_samples_split", payload.random_forest_min_samples_split)
        info.random_forest_max_features = report.get("random_forest_max_features", payload.random_forest_max_features)
        info.random_forest_bootstrap = report.get("random_forest_bootstrap", payload.random_forest_bootstrap)
        info.xgboost_n_estimators = report.get("xgboost_n_estimators", payload.xgboost_n_estimators)
        info.xgboost_max_depth = report.get("xgboost_max_depth", payload.xgboost_max_depth)
        info.xgboost_learning_rate = report.get("xgboost_learning_rate", payload.xgboost_learning_rate)
        info.xgboost_scale_pos_weight_auto = report.get("xgboost_scale_pos_weight_auto", payload.xgboost_scale_pos_weight_auto)
        info.xgboost_scale_pos_weight = report.get("xgboost_scale_pos_weight", payload.xgboost_scale_pos_weight)
        info.threshold_strategy = report.get("threshold_strategy", payload.threshold_strategy)
        info.target_recall = report.get("target_recall", payload.target_recall)
        info.false_negative_cost = report.get("false_negative_cost", payload.false_negative_cost)
        info.false_positive_cost = report.get("false_positive_cost", payload.false_positive_cost)
        info.experiment_hypothesis = report.get("experiment_hypothesis", payload.experiment_hypothesis)
        info.random_state = report.get("random_state", payload.random_state)
        info.best_model = report["best_model"]
        info.status = "success"
        LOGGER.info("Le run ML %s est terminé. Meilleur modèle : %s.", info.run_id, info.best_model)
    except Exception as error:  # noqa: BLE001 - on persiste l'erreur du run ML.
        info.status = "failed"
        info.error = str(error)
        append_ml_event(run_dir, "finish", "failed", f"Le run ML a échoué : {error}")
        LOGGER.error("Le run ML %s a échoué : %s", info.run_id, error)
        LOGGER.debug("Traceback du run ML:\n%s", traceback.format_exc())
    finally:
        info.finished_at = utc_now()
        write_ml_metadata(run_dir, info)
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()


def execute_vision_model_run(run_dir: Path, payload: VisionModelRunCreate) -> None:
    info = read_vision_model_metadata(run_dir)
    info.status = "running"
    info.started_at = utc_now()
    write_vision_model_metadata(run_dir, info)

    log_path = run_dir / "vision_training.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    try:
        manifest = load_latest_preparation(VISION_ARTIFACTS_DIR)
        if not manifest or manifest["version_id"] != info.dataset_version:
            raise ValueError("La version préparée du jeu de données a changé depuis la création du run.")
        LOGGER.info("Le run vision %s (%s) démarre sur %s.", info.run_id, payload.model_type, info.dataset_version)
        tracking_uri = f"sqlite:///{(VISION_MODEL_RUNS_DIR / 'mlflow.db').as_posix()}"
        if payload.model_type == "patchcore":
            report = train_patchcore(
                dataset_dir=MVTEC_DIR / "bottle", manifest=manifest, run_dir=run_dir, run_id=info.run_id,
                config=PatchCoreConfig(batch_size=payload.batch_size, threshold_percentile=payload.threshold_percentile, random_seed=payload.random_seed, coreset_ratio=payload.patchcore_coreset_ratio, max_memory_patches=payload.patchcore_max_memory_patches, candidate_patches=payload.patchcore_candidate_patches),
                mlflow_tracking_uri=tracking_uri,
            )
        else:
            report = train_vision_autoencoder(
                dataset_dir=MVTEC_DIR / "bottle", manifest=manifest, run_dir=run_dir, run_id=info.run_id,
                config=VisionTrainingConfig(epochs=payload.epochs, batch_size=payload.batch_size, learning_rate=payload.learning_rate, loss_name=payload.loss_name, latent_filters=payload.latent_filters, threshold_percentile=payload.threshold_percentile, early_stopping_patience=payload.early_stopping_patience, random_seed=payload.random_seed),
                mlflow_tracking_uri=tracking_uri,
            )
        info.status = "success"
        info.report_path = str((run_dir / "report.json").relative_to(PROJECT_DIR)).replace("\\", "/")
        LOGGER.info(
            "Run vision terminé : AUROC image=%s, AUROC pixel=%s.",
            report["metrics"]["image"]["auroc"],
            report["metrics"]["pixel"]["auroc"],
        )
    except Exception as error:  # noqa: BLE001 - l'erreur doit être visible depuis l'interface.
        info.status = "failed"
        info.error = str(error)
        LOGGER.error("Le run d'auto-encodeur %s a échoué : %s", info.run_id, error)
        LOGGER.debug("Traceback vision:\n%s", traceback.format_exc())
    finally:
        info.finished_at = utc_now()
        write_vision_model_metadata(run_dir, info)
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()


def read_existing_metadata(run_id: str) -> RunInfo:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run introuvable.")
    return read_metadata(run_dir)


def read_metadata(run_dir: Path) -> RunInfo:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata du run introuvable.")
    return RunInfo(**json.loads(metadata_path.read_text(encoding="utf-8")))


def read_existing_graph_metadata(run_id: str) -> GraphRunInfo:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run graphes introuvable.")
    return read_graph_metadata(run_dir)


def read_graph_metadata(run_dir: Path) -> GraphRunInfo:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata du run graphes introuvable.")
    return GraphRunInfo(**json.loads(metadata_path.read_text(encoding="utf-8")))


def read_existing_ml_metadata(run_id: str) -> MaintenanceMlRunInfo:
    run_dir = ML_RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run ML introuvable.")
    return read_ml_metadata(run_dir)


def read_ml_metadata(run_dir: Path) -> MaintenanceMlRunInfo:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Métadonnées du run ML introuvables.")
    return MaintenanceMlRunInfo(**json.loads(metadata_path.read_text(encoding="utf-8")))


def resolve_vision_model_run_dir(run_id: str) -> Path:
    if not (run_id.startswith("vision_ae_") or run_id.startswith("vision_patchcore_")) or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="Identifiant de run de vision non valide.")
    run_dir = VISION_MODEL_RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run de vision introuvable.")
    return run_dir


def read_vision_model_metadata(run_dir: Path) -> VisionModelRunInfo:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Métadonnée du run de vision introuvable.")
    return VisionModelRunInfo(**json.loads(metadata_path.read_text(encoding="utf-8")))


def vision_split(dataset_dir: Path, relative_dir: str, label: str) -> VisionDatasetSplit:
    split_dir = dataset_dir / relative_dir
    files = sorted(split_dir.glob("*.png")) if split_dir.exists() else []
    return VisionDatasetSplit(
        name=relative_dir.replace("\\", "/"),
        label=label,
        count=len(files),
        sample_paths=[str(path.relative_to(dataset_dir)).replace("\\", "/") for path in files[:4]],
    )


def resolve_vision_image_path(path: str) -> Path:
    if "\\" in path or path.startswith("/") or ".." in Path(path).parts:
        raise HTTPException(status_code=400, detail="Chemin image invalide.")
    dataset_dir = (MVTEC_DIR / "bottle").resolve()
    image_path = (dataset_dir / path).resolve()
    if not image_path.is_relative_to(dataset_dir) or image_path.suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Chemin image invalide.")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image introuvable.")
    return image_path


def resolve_gold_csv_path(run_name: str) -> Path:
    if "/" in run_name or "\\" in run_name or ".." in run_name:
        raise HTTPException(status_code=400, detail="Nom de run Gold invalide.")
    run_dir = GOLD_DIR / run_name
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run Gold introuvable.")
    csv_files = sorted(run_dir.glob("gold_dataset_*.csv"), reverse=True)
    if not csv_files:
        raise HTTPException(status_code=404, detail="CSV Gold introuvable pour ce run.")
    return csv_files[0]


def write_metadata(run_dir: Path, info: RunInfo) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(info.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_ml_metadata(run_dir: Path, info: MaintenanceMlRunInfo) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(info.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_vision_model_metadata(run_dir: Path, info: VisionModelRunInfo) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(info.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_ml_event(run_dir: Path, step: str, status: str, message: str, **details: Any) -> None:
    event = {
        "ts": utc_now(),
        "step": step,
        "status": status,
        "message": message,
        "details": details,
    }
    with (run_dir / "training_events.jsonl").open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    records = []
    for record in cleaned.to_dict(orient="records"):
        records.append({key: json_safe_value(value) for key, value in record.items()})
    return records


def json_safe_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def optional_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def mlflow_report_fallbacks() -> dict[tuple[str | None, str | None, float | None], dict[str, Any]]:
    fallbacks: dict[tuple[str | None, str | None, float | None], dict[str, Any]] = {}
    dataset_hash_cache: dict[str, str | None] = {}
    for report_path in ML_RUNS_DIR.glob("maintenance_ml_*/report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        gold_csv = optional_str(report.get("gold_csv_path"))
        dataset_hash = (report.get("reproducibility") or {}).get("dataset_hash")
        if not dataset_hash and gold_csv:
            dataset_hash = dataset_hash_cache.setdefault(gold_csv, compute_csv_dataset_hash(gold_csv))
        for result in report.get("results", []):
            key = mlflow_fallback_key(gold_csv, optional_str(result.get("model")), optional_float(result.get("threshold")))
            business_cost = result.get("business_cost_test")
            if business_cost is None:
                matrix = result.get("test_confusion_matrix") or {}
                business_cost = (
                    float(matrix.get("fn", 0)) * float(report.get("false_negative_cost", 20))
                    + float(matrix.get("fp", 0)) * float(report.get("false_positive_cost", 1))
                )
            fallbacks[key] = {
                "app_run_id": report.get("run_id"),
                "dataset_hash": dataset_hash,
                "threshold_strategy": report.get("threshold_strategy"),
                "test_business_cost": business_cost,
            }
    return fallbacks


def mlflow_fallback_key(gold_csv: str | None, model: str | None, threshold: float | None) -> tuple[str | None, str | None, float | None]:
    return (normalize_path_key(gold_csv), model, round(threshold, 12) if threshold is not None else None)


def normalize_path_key(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().as_posix().lower()
    except Exception:
        return path.replace("\\", "/").lower()


def compute_csv_dataset_hash(csv_path: str) -> str | None:
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        return hash_dataframe(pd.read_csv(path))
    except Exception as error:
        LOGGER.warning("Hash dataset impossible pour %s : %s", csv_path, error)
        return None


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def tail_text(path: Path, max_chars: int = 1200) -> str:
    if not path.exists():
        return "aucun log MLflow UI disponible"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return "log MLflow UI vide"
    return text[-max_chars:]


def write_candidate_readme(run_id: str, report: dict[str, Any], best: dict[str, Any], version: str, stage: str) -> Path:
    run_dir = ML_RUNS_DIR / run_id
    reproducibility = report.get("reproducibility") or {}
    readme = f"""# Modèle candidat maintenance prédictive

## Identité

- Run applicatif : `{run_id}`
- Modèle : `{best.get("model", report.get("best_model"))}`
- Registry : `{MLFLOW_REGISTERED_MODEL_NAME}`
- Version : `{version}`
- Stage : `{stage}`
- MLflow run : `{best.get("mlflow_run_id", "-")}`

## Données

- Gold dataset : `{report.get("gold_run_name", "-")}`
- CSV : `{report.get("gold_csv_path", "-")}`
- Hash dataset : `{reproducibility.get("dataset_hash", "-")}`
- Cible : `{report.get("label_column", "-")}`
- Lignes : `{report.get("rows", "-")}`
- Features : `{report.get("features", "-")}`

## Décision opérationnelle

- Stratégie de seuil : `{report.get("threshold_strategy", "-")}`
- Seuil retenu : `{best.get("threshold", "-")}`
- Coût métier : `FN x{report.get("false_negative_cost", 20)} / FP x{report.get("false_positive_cost", 1)}`

## Métriques test

- PR-AUC : `{best.get("pr_auc_test", "-")}`
- ROC-AUC : `{best.get("roc_auc_test", "-")}`
- Précision : `{best.get("precision_test", "-")}`
- Recall : `{best.get("recall_test", "-")}`
- F1 : `{best.get("f1_test", "-")}`
- Coût métier : `{best.get("business_cost_test", "-")}`

## Reproductibilité

- Seed : `{report.get("random_state", 42)}`
- Python : `{reproducibility.get("python_version", "-")}`
- scikit-learn : `{reproducibility.get("sklearn_version", "-")}`
- Tracking URI : `{report.get("mlflow_tracking_uri", "-")}`

## Limites

- Le seuil est choisi sur validation et doit être surveillé en production.
- SHAP global et local est disponible dans les artefacts quand la librairie SHAP est installée.
- Le coût métier dépend des pondérations FN/FP configurées au run.
"""
    readme_path = run_dir / "model_candidate_README.md"
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def load_latest_success_report_for_registered_model() -> dict[str, Any] | None:
    for report_path in sorted(ML_RUNS_DIR.glob("maintenance_ml_*/report.json"), reverse=True):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if report.get("status") == "success":
            return report
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
