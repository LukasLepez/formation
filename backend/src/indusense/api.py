"""Backend FastAPI pour executer les pipelines InduSense."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from indusense.artifacts import RUN_INDEX_JSON
from indusense.processing.ingestion import (
    DEFAULT_DATABASE_URL,
    GoldDatasetConfig,
    ensure_postgres_stack_running,
    run_layer_pipeline,
)
from indusense.reporting.graphs import generate_graph_report


PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_DIR / "artifacts" / "pipeline-runs"
GOLD_DIR = PROJECT_DIR / "artifacts" / "gold-datasets"
LOGGER = logging.getLogger(__name__)
DOCKER_START_LOCK = threading.Lock()

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
