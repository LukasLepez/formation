"""Constructeur du Gold Dataset pour le projet de formation InduSense 4.0.

L'implémentation suit ``docs/gold_dataset.md`` comme contrat :
une ligne représente une machine pendant une heure, toutes les variables
glissantes sont calculées par machine, et toute statistique apprise depuis les
données est ajustée uniquement sur le split chronologique d'entraînement.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv

from indusense.artifacts import INCIDENT_ARTIFACT_ROOT, project_relative, update_incident_run_indexes


LOGGER = logging.getLogger(__name__)
POSTGRES_CONTAINER = "formation-postgres"
PGADMIN_CONTAINER = "formation-pgadmin"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_DB = "formation_indusense"
POSTGRES_PORT = "5432"
PGADMIN_URL = "http://localhost:5050"
DEFAULT_DATABASE_URL = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
)

SIGNAL_COLUMNS = {
    "temp": "temp_mean_1h",
    "pressure": "pressure_mean_1h",
    "voltage": "voltage_mean_1h",
    "rotation": "rotation_mean_1h",
}
TYPE_COLUMNS = [
    "type_surchauffe",
    "type_baisse_pression",
    "type_vibration",
    "type_bruit_mecanique",
    "type_surconsommation",
    "type_blocage_mecanique",
    "type_alarme_capteur",
    "type_arret_urgence",
    "type_defaut_qualite",
]
BRONZE_TELEMETRY_COLUMNS = [
    "machine_id",
    "timestamp",
    "temperature_c",
    "pressure_bar",
    "voltage_mean_v",
    "rotation_mean_rpm",
    "pieces_produced",
    "ingested_at",
]
BRONZE_INCIDENT_COLUMNS = [
    "incident_id",
    "date",
    "time",
    "operator_key",
    "machine_id",
    "severity",
    "comment",
    "shift",
    *TYPE_COLUMNS,
    "ingested_at",
]
MACHINE_COLUMNS = [
    "machine_code",
    "commissioning_date",
    "max_daily_capacity",
    "max_hourly_capacity_pieces",
    "model",
    "production_line",
    "location",
    "criticality",
]
MAINTENANCE_COLUMNS = [
    "maintenance_id",
    "machine_code",
    "maintenance_at",
    "maintenance_type",
    "action_type",
    "component",
    "description",
    "related_incident_id",
    "duration_hours",
]


@dataclass(frozen=True)
class GoldDatasetConfig:
    """Chemins d'exécution et options de persistance du pipeline Gold."""

    telemetry_path: Path = Path("data/telemetry.csv")
    incidents_path: Path = Path("data/releves_incidents.csv")
    machine_sql_path: Path = Path("data/machine.sql")
    output_dir: Path = Path("artifacts/gold-datasets")
    persist_db: bool = True
    database_url: str | None = None
    table_name: str = "gold_dataset"
    auto_start_docker: bool = True
    export_google_sheets: bool = True
    google_sheets_max_cells: int = 9_500_000
    layer: str = "all"


def configure_logging(level: str = "INFO") -> None:
    """Configure des logs console lisibles pour les exécutions locales et en ligne de commande."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def load_gold_from_db(
    database_url: str | None = None,
    table_name: str = "gold_dataset",
    schema: str = "gold",
) -> pd.DataFrame:
    """Recharge le Gold Dataset canonique depuis PostgreSQL.

    ``DATABASE_URL`` est utilisée quand ``database_url`` n'est pas fourni. Cela
    évite d'écrire des identifiants en dur dans le dépôt.
    """

    load_dotenv()
    url = database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL

    LOGGER.info("Chargement du Gold Dataset depuis PostgreSQL dans le schéma « %s » et la table « %s ».", schema, table_name)
    stored = pd.read_sql_table(table_name, create_engine(url), schema=schema)
    if {"features", "labels"}.issubset(stored.columns):
        return expand_gold_storage_frame(stored)
    return order_gold_columns(stored)


def create_database_engine(config: GoldDatasetConfig):
    """Crée le moteur SQLAlchemy configuré pour la base locale InduSense."""

    database_url = config.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    return create_engine(database_url, future=True)


def prepare_database(config: GoldDatasetConfig):
    """Démarre PostgreSQL si besoin, applique Alembic et renvoie le moteur SQLAlchemy."""

    if config.auto_start_docker:
        ensure_postgres_stack_running()
    engine = create_database_engine(config)
    run_alembic_upgrade(config)
    repair_database_schema(engine)
    return engine


def run_alembic_upgrade(config: GoldDatasetConfig) -> None:
    """Applique les migrations Alembic jusqu'à la dernière version."""

    os.environ["DATABASE_URL"] = config.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    alembic_ini = find_project_dir(Path.cwd()) / "alembic.ini"
    alembic_config = Config(str(alembic_ini))
    LOGGER.info("Application des migrations Alembic avec le fichier %s.", alembic_ini)
    engine = create_database_engine(config)
    if should_stamp_existing_database(engine):
        LOGGER.warning(
            "Des tables InduSense existent déjà sans révision Alembic enregistrée. "
            "Synchronisation du schéma existant, puis marquage de la migration comme appliquée."
        )
        synchronize_existing_database_schema(engine)
        run_alembic_command_with_logging_preserved(command.stamp, alembic_config, "head")
        return
    run_alembic_command_with_logging_preserved(command.upgrade, alembic_config, "head")


def run_alembic_command_with_logging_preserved(alembic_command, *args) -> None:
    """Execute Alembic sans perdre les handlers de logs du backend."""

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    try:
        alembic_command(*args)
    finally:
        for handler in previous_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
        root_logger.setLevel(previous_level)


def should_stamp_existing_database(engine) -> bool:
    """Detecte une base creee avant Alembic ou par un run interrompu."""

    inspector = inspect(engine)
    current_revision = get_current_alembic_revision(engine, inspector)
    if current_revision:
        return False
    managed_tables = [
        ("bronze", "telemetry_raw"),
        ("bronze", "incidents_raw"),
        ("bronze", "machine"),
        ("bronze", "maintenance"),
        ("silver", "telemetry"),
        ("silver", "incidents"),
        ("silver", "machine"),
        ("silver", "maintenance"),
        ("gold", "gold_dataset"),
    ]
    return any(inspector.has_table(table, schema=schema) for schema, table in managed_tables)


def get_current_alembic_revision(engine, inspector) -> str | None:
    """Retourne la revision Alembic courante si elle existe."""

    if not inspector.has_table("alembic_version"):
        return None
    with engine.begin() as connection:
        return connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).scalar()


def synchronize_existing_database_schema(engine) -> None:
    """Complete les tables manquantes sans toucher aux tables existantes."""

    from indusense.db.models import Base

    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS silver"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
    Base.metadata.create_all(engine, checkfirst=True)


def repair_database_schema(engine) -> None:
    """Ajoute les colonnes attendues quand une table existante vient d'un ancien schema."""

    LOGGER.info("Vérification des colonnes de compatibilité attendues par les couches Bronze, Silver et Gold.")
    statements = [
        "ALTER TABLE bronze.telemetry_raw ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP NOT NULL DEFAULT now()",
        "ALTER TABLE bronze.incidents_raw ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP NOT NULL DEFAULT now()",
        "ALTER TABLE silver.telemetry ADD COLUMN IF NOT EXISTS machine_id VARCHAR(16)",
        "ALTER TABLE silver.telemetry ADD COLUMN IF NOT EXISTS machine_id_std VARCHAR(16)",
        "ALTER TABLE silver.telemetry ADD COLUMN IF NOT EXISTS window_start TIMESTAMP",
        "ALTER TABLE silver.incidents ADD COLUMN IF NOT EXISTS incident_at TIMESTAMP",
        "ALTER TABLE silver.incidents ADD COLUMN IF NOT EXISTS window_start TIMESTAMP",
        "ALTER TABLE silver.incidents ADD COLUMN IF NOT EXISTS machine_id_std VARCHAR(16)",
        "ALTER TABLE silver.maintenance ADD COLUMN IF NOT EXISTS maintenance_hour TIMESTAMP",
    ]
    with engine.begin() as connection:
        for schema in ("bronze", "silver", "gold"):
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        for statement in statements:
            connection.execute(text(statement))


def reset_layer_tables(engine, layer: str) -> None:
    """Recree les tables de la couche cible pour eviter les derives de vieux schemas."""

    LOGGER.info("Recréation des tables de la couche « %s » pour repartir sur le schéma actuel.", layer)
    from indusense.db.models import Base

    layer_tables = {
        "bronze": ["telemetry_raw", "incidents_raw", "machine", "maintenance"],
        "silver": ["telemetry", "incidents", "machine", "maintenance"],
        "gold": ["gold_dataset"],
    }
    if layer not in layer_tables:
        raise ValueError("layer doit valoir bronze, silver ou gold")

    table_names = layer_tables[layer]
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {layer}"))
        for table_name in table_names:
            LOGGER.info("Suppression de la table %s.%s si elle existe déjà.", layer, table_name)
            connection.execute(text(f'DROP TABLE IF EXISTS "{layer}"."{table_name}" CASCADE'))

    tables = [Base.metadata.tables[f"{layer}.{table_name}"] for table_name in table_names]
    Base.metadata.create_all(engine, tables=tables, checkfirst=True)
    LOGGER.info("Les tables de la couche « %s » ont été recréées.", layer)


def truncate_table(engine, schema: str, table_name: str) -> None:
    """Vide une table en réinitialisant ses identifiants techniques."""

    with engine.begin() as connection:
        connection.execute(text(f'TRUNCATE TABLE "{schema}"."{table_name}" RESTART IDENTITY CASCADE'))


def write_table(
    df: pd.DataFrame,
    engine,
    schema: str,
    table_name: str,
    columns: list[str] | None = None,
) -> None:
    """Écrit un DataFrame dans une table existante en conservant la structure Alembic."""

    data = df.copy()
    if columns is not None:
        data = data[columns].copy()
    truncate_table(engine, schema, table_name)
    dtype = {}
    if "features" in data.columns:
        dtype["features"] = JSONB()
    if "labels" in data.columns:
        dtype["labels"] = JSONB()
    data.to_sql(table_name, engine, schema=schema, if_exists="append", index=False, chunksize=1000, dtype=dtype or None)
    LOGGER.info("Écriture de %s lignes dans la table %s.%s.", f"{len(data):,}", schema, table_name)


def read_table(engine, schema: str, table_name: str) -> pd.DataFrame:
    """Lit une table qualifiée par schéma sans dépendre du search_path PostgreSQL."""

    return pd.read_sql_query(text(f'SELECT * FROM "{schema}"."{table_name}"'), engine)


def write_layer_run_metadata(layer: str, frames: dict[str, pd.DataFrame], source: str) -> Path:
    """Enregistre un run Bronze/Silver dans les artefacts historiques."""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{timestamp}_{layer}"
    run_dir = INCIDENT_ARTIFACT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / f"rapport_pipeline_{layer}.md"
    metadata_path = run_dir / "metadata.json"

    telemetry = frames.get("telemetry", pd.DataFrame())
    incidents = frames.get("incidents", pd.DataFrame())
    machines = frames.get("machines", pd.DataFrame())
    maintenance = frames.get("maintenance", pd.DataFrame())
    machine_column = "machine_id_std" if "machine_id_std" in telemetry else "machine_id"
    machines_uniques = int(telemetry[machine_column].nunique()) if machine_column in telemetry else 0

    metadata = {
        "run_ts": timestamp,
        "run_name": run_name,
        "layer": layer,
        "source_layer": source,
        "schema": layer,
        "run_dir": project_relative(run_dir),
        "report_path": project_relative(report_path),
        "nombre_graphes": 0,
        "nombre_lignes": int(len(incidents)),
        "nombre_lignes_telemetrie_lues": int(len(telemetry)),
        "nombre_lignes_telemetrie_utilisees": int(len(telemetry.dropna(subset=["timestamp"]))) if "timestamp" in telemetry else int(len(telemetry)),
        "machines_uniques": machines_uniques,
        "nombre_machines_reference": int(len(machines)),
        "nombre_maintenances": int(len(maintenance)),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                f"# Run pipeline {layer}",
                "",
                f"- Run : `{run_name}`",
                f"- Source : `{source}`",
                f"- Incidents : {metadata['nombre_lignes']:,}",
                f"- Telemetrie : {metadata['nombre_lignes_telemetrie_lues']:,}",
                f"- Machines : {metadata['machines_uniques']:,}",
                f"- Maintenances : {metadata['nombre_maintenances']:,}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    update_incident_run_indexes(metadata)
    LOGGER.info("Les artefacts du run « %s » ont été enregistrés dans le dossier %s.", layer, run_dir)
    return run_dir


def build_bronze_layer(config: GoldDatasetConfig, engine) -> dict[str, pd.DataFrame]:
    """Charge les sources brutes et les persiste dans le schéma bronze."""

    LOGGER.info("Démarrage de la couche Bronze : lecture des sources et chargement dans PostgreSQL.")
    telemetry_raw = load_telemetry(config.telemetry_path)
    incidents_raw = load_incidents(config.incidents_path)
    machines, maintenance = load_machine_reference(config.machine_sql_path)
    ingested_at = datetime.now().replace(microsecond=0)
    telemetry_raw["ingested_at"] = ingested_at
    incidents_raw["ingested_at"] = ingested_at

    reset_layer_tables(engine, "bronze")
    write_table(telemetry_raw, engine, "bronze", "telemetry_raw", BRONZE_TELEMETRY_COLUMNS)
    write_table(incidents_raw, engine, "bronze", "incidents_raw", BRONZE_INCIDENT_COLUMNS)
    write_table(machines, engine, "bronze", "machine", MACHINE_COLUMNS)
    write_table(maintenance, engine, "bronze", "maintenance", MAINTENANCE_COLUMNS)
    write_layer_run_metadata(
        "bronze",
        {
            "telemetry": telemetry_raw,
            "incidents": incidents_raw,
            "machines": machines,
            "maintenance": maintenance,
        },
        source="files",
    )
    return {
        "telemetry": telemetry_raw,
        "incidents": incidents_raw,
        "machines": machines,
        "maintenance": maintenance,
    }


def build_silver_layer(config: GoldDatasetConfig, engine) -> dict[str, pd.DataFrame]:
    """Construit Silver depuis les tables Bronze déjà présentes en base."""

    LOGGER.info("Démarrage de la couche Silver : lecture de Bronze, nettoyage des données et persistance.")
    telemetry_raw = read_table(engine, "bronze", "telemetry_raw")
    incidents_raw = read_table(engine, "bronze", "incidents_raw")
    machines = read_table(engine, "bronze", "machine")
    maintenance = read_table(engine, "bronze", "maintenance")

    telemetry_raw["timestamp"] = pd.to_datetime(telemetry_raw["timestamp"], errors="coerce")
    telemetry_silver = build_silver_telemetry(telemetry_raw)
    incidents_silver = prepare_silver_incidents(incidents_raw)
    maintenance_silver = prepare_silver_maintenance(maintenance)

    reset_layer_tables(engine, "silver")
    write_table(
        telemetry_silver,
        engine,
        "silver",
        "telemetry",
        [
            "machine_id",
            "timestamp",
            "temperature_c",
            "pressure_bar",
            "voltage_mean_v",
            "rotation_mean_rpm",
            "pieces_produced",
            "machine_id_std",
            "window_start",
        ],
    )
    write_table(
        incidents_silver,
        engine,
        "silver",
        "incidents",
        [
            "incident_id",
            "date",
            "time",
            "operator_key",
            "machine_id",
            "severity",
            "comment",
            "shift",
            "incident_at",
            "window_start",
            "machine_id_std",
            *TYPE_COLUMNS,
        ],
    )
    write_table(machines, engine, "silver", "machine", MACHINE_COLUMNS)
    write_table(maintenance_silver, engine, "silver", "maintenance", [*MAINTENANCE_COLUMNS, "maintenance_hour"])
    write_layer_run_metadata(
        "silver",
        {
            "telemetry": telemetry_silver,
            "incidents": incidents_silver,
            "machines": machines,
            "maintenance": maintenance_silver,
        },
        source="bronze",
    )
    return {
        "telemetry": telemetry_silver,
        "incidents": incidents_silver,
        "machines": machines,
        "maintenance": maintenance_silver,
    }


def build_gold_layer(config: GoldDatasetConfig, engine) -> pd.DataFrame:
    """Construit Gold depuis Silver, puis persiste CSV et table Gold canonique."""

    LOGGER.info("Démarrage de la couche Gold : lecture de Silver, puis construction des variables explicatives et des labels.")
    telemetry_silver = read_table(engine, "silver", "telemetry")
    incidents_silver = read_table(engine, "silver", "incidents")
    machines = read_table(engine, "silver", "machine")
    maintenance = read_table(engine, "silver", "maintenance")

    telemetry_silver["timestamp"] = pd.to_datetime(telemetry_silver["timestamp"], errors="coerce")
    telemetry_silver["window_start"] = pd.to_datetime(telemetry_silver["window_start"], errors="coerce")
    incidents_silver["incident_at"] = pd.to_datetime(incidents_silver["incident_at"], errors="coerce")
    incidents_silver["window_start"] = pd.to_datetime(incidents_silver["window_start"], errors="coerce")
    maintenance["maintenance_at"] = pd.to_datetime(maintenance["maintenance_at"], errors="coerce")
    maintenance["maintenance_hour"] = pd.to_datetime(maintenance["maintenance_hour"], errors="coerce")

    gold = build_gold_from_silver_frames(telemetry_silver, incidents_silver, machines, maintenance)
    validate_gold_dataset(gold)
    persist_gold_dataset(gold, config, engine=engine)
    LOGGER.info("La couche Gold est terminée : %s lignes et %s colonnes ont été produites.", f"{len(gold):,}", gold.shape[1])
    return gold


def run_layer_pipeline(config: GoldDatasetConfig | None = None) -> pd.DataFrame | None:
    """Exécute bronze, silver, gold ou les trois couches selon la configuration."""

    load_dotenv()
    config = config or GoldDatasetConfig(database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL)
    layer = config.layer.lower()
    if layer not in {"bronze", "silver", "gold", "all"}:
        raise ValueError("layer doit valoir bronze, silver, gold ou all")
    if not config.persist_db and layer != "gold":
        raise ValueError("--no-db est seulement compatible avec --layer gold.")

    if not config.persist_db:
        return build_gold_from_telemetry(config)

    engine = prepare_database(config)
    if layer in {"bronze", "all"}:
        build_bronze_layer(config, engine)
    if layer in {"silver", "all"}:
        build_silver_layer(config, engine)
    if layer in {"gold", "all"}:
        return build_gold_layer(config, engine)
    return None


def build_gold_from_telemetry(config: GoldDatasetConfig | None = None) -> pd.DataFrame:
    """Construit et persiste éventuellement le Gold Dataset complet.

    La fonction est volontairement bavarde dans ses logs, car ce projet est
    pédagogique : chaque transformation majeure annonce son volume d'entrée,
    son volume de sortie et les garde-fous contre la fuite de données.
    """

    load_dotenv()
    config = config or GoldDatasetConfig(database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL)
    LOGGER.info("Démarrage du pipeline Gold InduSense 4.0.")
    LOGGER.info("La pipeline utilise le fichier de télémétrie %s, le fichier des incidents %s et le référentiel machine %s.", config.telemetry_path, config.incidents_path, config.machine_sql_path)

    telemetry_raw = load_telemetry(config.telemetry_path)
    incidents_raw = load_incidents(config.incidents_path)
    machines, maintenance = load_machine_reference(config.machine_sql_path)

    telemetry_silver = build_silver_telemetry(telemetry_raw)
    incidents_silver = prepare_silver_incidents(incidents_raw)
    maintenance_silver = prepare_silver_maintenance(maintenance)
    gold = build_gold_from_silver_frames(telemetry_silver, incidents_silver, machines, maintenance_silver)

    validate_gold_dataset(gold)
    persist_gold_dataset(gold, config)
    LOGGER.info("Le pipeline Gold est terminé : %s lignes et %s colonnes ont été produites.", f"{len(gold):,}", gold.shape[1])
    return gold


def build_gold_from_silver_frames(
    telemetry_silver: pd.DataFrame,
    incidents_silver: pd.DataFrame,
    machines: pd.DataFrame,
    maintenance: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble le DataFrame Gold depuis les tables Silver déjà nettoyées."""

    hourly = aggregate_hourly(telemetry_silver)
    hourly = complete_hourly_grid(hourly)
    hourly = add_temporal_split(hourly)
    hourly = impute_hourly_signals_train_only(hourly)
    hourly = add_machine_reference(hourly, machines)
    hourly = add_incident_context(hourly, incidents_silver)
    hourly = add_maintenance_context(hourly, maintenance)
    hourly = add_memory_features(hourly)
    hourly = add_trend_features(hourly)
    hourly = add_anomaly_features(hourly)
    hourly = add_production_features(hourly)
    gold = add_future_labels(hourly)
    return order_gold_columns(gold)


def load_telemetry(path: Path) -> pd.DataFrame:
    """Charge la télémétrie brute et vérifie les colonnes physiques attendues."""

    LOGGER.info("Lecture du fichier de télémétrie Bronze : %s.", path)
    df = pd.read_csv(path)
    required = {"machine_id", "timestamp", "temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm", "pieces_produced"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes de télémétrie manquantes : {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in required - {"machine_id", "timestamp"}:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    LOGGER.info("La télémétrie Bronze contient %s lignes, %s machines et %s horodatages invalides.", f"{len(df):,}", df["machine_id"].nunique(), df["timestamp"].isna().sum())
    return df


def load_incidents(path: Path) -> pd.DataFrame:
    """Charge les incidents opérateur et prépare un timestamp horaire pour les jointures."""

    LOGGER.info("Lecture du fichier des incidents Bronze : %s.", path)
    df = pd.read_csv(path)
    required = {"incident_id", "date", "time", "machine_id", "severity", *TYPE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes d'incidents manquantes : {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["time"] = parse_time_column(df["time"])
    df["incident_at"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df["window_start"] = df["incident_at"].dt.floor("h")
    df["machine_id_std"] = df["machine_id"].astype(str).str.strip().str.upper()
    df = anonymize_incident_operators(df)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
    for column in TYPE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    LOGGER.info("Les incidents Bronze contiennent %s lignes et %s horodatages invalides.", f"{len(df):,}", df["incident_at"].isna().sum())
    return df


def prepare_silver_incidents(raw: pd.DataFrame) -> pd.DataFrame:
    """Ajoute aux incidents Bronze les clés horaires utilisées par Silver et Gold."""

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["time"] = parse_time_column(df["time"])
    df["incident_at"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df["window_start"] = df["incident_at"].dt.floor("h")
    df["machine_id_std"] = df["machine_id"].astype(str).str.strip().str.upper()
    df = anonymize_incident_operators(df)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0).astype(int)
    for column in TYPE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    invalid = df["incident_at"].isna().sum()
    if invalid:
        raise ValueError(f"Incidents Silver invalides: {invalid} horodatages non parsables")
    return df


def anonymize_incident_operators(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les identifiants opérateurs directs par une clé anonyme non réversible."""

    anonymized = df.copy()
    if "operator_key" in anonymized.columns:
        return anonymized

    if "operator_badge" in anonymized.columns:
        unique_badges = sorted(anonymized["operator_badge"].dropna().astype(str).unique())
        operator_keys = {badge: f"OP_{secrets.token_urlsafe(12)}" for badge in unique_badges}
        anonymized["operator_key"] = anonymized["operator_badge"].astype(str).map(operator_keys)
    else:
        anonymized["operator_key"] = [f"OP_{secrets.token_urlsafe(12)}" for _ in range(len(anonymized))]

    if anonymized["operator_key"].isna().any():
        raise ValueError("Certains incidents n'ont pas pu être reliés à un opérateur anonymisé.")

    return anonymized.drop(columns=["operator_name", "operator_badge"], errors="ignore")


def parse_time_column(values: pd.Series) -> pd.Series:
    """Parse les heures source au format HH:MM ou HH:MM:SS."""

    as_text = values.astype(str)
    parsed = pd.to_datetime(as_text, format="%H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(as_text.loc[missing], format="%H:%M:%S", errors="coerce")
    return parsed.dt.time


def load_machine_reference(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrait les référentiels machine et maintenance depuis le fichier SQL source."""

    LOGGER.info("Lecture du référentiel machine et maintenance depuis le fichier SQL : %s.", path)
    sql = path.read_text(encoding="utf-8")
    machine_columns = [
        "machine_code",
        "commissioning_date",
        "max_daily_capacity",
        "max_hourly_capacity_pieces",
        "model",
        "production_line",
        "location",
        "criticality",
    ]
    maintenance_columns = [
        "maintenance_id",
        "machine_code",
        "maintenance_at",
        "maintenance_type",
        "action_type",
        "component",
        "description",
        "related_incident_id",
        "duration_hours",
    ]
    machines = _extract_insert_values(sql, "machine", machine_columns)
    maintenance = _extract_insert_values(sql, "maintenance", maintenance_columns)

    machines["commissioning_date"] = pd.to_datetime(machines["commissioning_date"], errors="coerce").dt.date
    for column in ["max_daily_capacity", "max_hourly_capacity_pieces"]:
        machines[column] = pd.to_numeric(machines[column], errors="coerce")
    maintenance = prepare_silver_maintenance(maintenance)
    maintenance["duration_hours"] = pd.to_numeric(maintenance["duration_hours"], errors="coerce")
    LOGGER.info("Le référentiel Bronze contient %s machines et %s maintenances.", len(machines), f"{len(maintenance):,}")
    return machines, maintenance


def prepare_silver_maintenance(raw: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la fenêtre horaire de maintenance utilisée par les features Gold."""

    df = raw.copy()
    df["maintenance_at"] = pd.to_datetime(df["maintenance_at"], errors="coerce", utc=True).dt.tz_convert(None)
    df["maintenance_hour"] = df["maintenance_at"].dt.floor("h")
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    return df


def _extract_insert_values(sql: str, table: str, columns: list[str]) -> pd.DataFrame:
    """Transforme un bloc PostgreSQL ``INSERT ... VALUES`` en DataFrame."""

    pattern = rf"INSERT INTO {table} \([^)]+\)\s*VALUES\s*(.*?)ON CONFLICT"
    match = re.search(pattern, sql, flags=re.S | re.I)
    if not match:
        raise ValueError(f"Bloc INSERT introuvable pour la table {table!r}")

    rows = []
    for raw_tuple in re.findall(r"\((.*?)\)(?:,|$)", match.group(1), flags=re.S):
        python_tuple = raw_tuple.replace("NULL", "None")
        rows.append(ast.literal_eval(f"({python_tuple})"))
    return pd.DataFrame(rows, columns=columns)


def build_silver_telemetry(raw: pd.DataFrame) -> pd.DataFrame:
    """Déduplique, standardise les identifiants et convertit les valeurs Fahrenheit suspectes."""

    LOGGER.info("Début du nettoyage Silver de la télémétrie avec %s lignes en entrée.", f"{len(raw):,}")
    df = raw.copy()
    df["machine_id_std"] = df["machine_id"].astype(str).str.strip().str.upper()
    df["window_start"] = df["timestamp"].dt.floor("h")

    duplicate_mask = df.duplicated(["machine_id_std", "timestamp"], keep="first")
    LOGGER.info("%s doublons exacts machine et horodatage ont été détectés dans la télémétrie Silver.", f"{duplicate_mask.sum():,}")
    df = df.loc[~duplicate_mask].copy()

    fahrenheit_mask = df["temperature_c"] > 80
    LOGGER.info("%s températures suspectes en Fahrenheit ont été converties en Celsius.", f"{fahrenheit_mask.sum():,}")
    df.loc[fahrenheit_mask, "temperature_c"] = (df.loc[fahrenheit_mask, "temperature_c"] - 32) * 5 / 9

    invalid_timestamps = df["timestamp"].isna().sum()
    if invalid_timestamps:
        LOGGER.warning("%s lignes de télémétrie Silver avec un horodatage invalide ont été supprimées.", invalid_timestamps)
        df = df.dropna(subset=["timestamp"])
    LOGGER.info("Le nettoyage Silver de la télémétrie est terminé avec %s lignes conservées.", f"{len(df):,}")
    return df


def aggregate_hourly(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Agrège la télémétrie brute en une ligne observée par machine et par heure."""

    LOGGER.info("Agrégation Gold : regroupement de la télémétrie par machine et par heure.")
    hourly = (
        telemetry.groupby(["machine_id_std", "window_start"], as_index=False)
        .agg(
            temp_mean_1h=("temperature_c", "mean"),
            temp_max_1h=("temperature_c", "max"),
            pressure_mean_1h=("pressure_bar", "mean"),
            pressure_max_1h=("pressure_bar", "max"),
            voltage_mean_1h=("voltage_mean_v", "mean"),
            voltage_max_1h=("voltage_mean_v", "max"),
            rotation_mean_1h=("rotation_mean_rpm", "mean"),
            rotation_max_1h=("rotation_mean_rpm", "max"),
            pieces_produced_1h=("pieces_produced", "sum"),
        )
        .sort_values(["machine_id_std", "window_start"])
        .reset_index(drop=True)
    )
    LOGGER.info("L'agrégation horaire Gold a produit %s lignes pour %s machines.", f"{len(hourly):,}", hourly["machine_id_std"].nunique())
    return hourly


def complete_hourly_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    """Crée toutes les lignes machine-heure attendues pour permettre l'imputation."""

    LOGGER.info("Création de la grille horaire Gold complète pour chaque machine.")
    machines = sorted(hourly["machine_id_std"].unique())
    full_index = []
    for machine, group in hourly.groupby("machine_id_std", sort=True):
        hours = pd.date_range(group["window_start"].min(), group["window_start"].max(), freq="h")
        full_index.extend((machine, hour) for hour in hours)

    grid = pd.DataFrame(full_index, columns=["machine_id_std", "window_start"])
    completed = grid.merge(hourly, on=["machine_id_std", "window_start"], how="left")
    LOGGER.info("La grille horaire Gold couvre %s machines : %s lignes observées, %s lignes complètes et %s lignes ajoutées.", len(machines), f"{len(hourly):,}", f"{len(completed):,}", f"{len(completed) - len(hourly):,}")
    return completed.sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)


def add_temporal_split(hourly: pd.DataFrame) -> pd.DataFrame:
    """Attribue entraînement/validation/test uniquement par quantiles chronologiques."""

    LOGGER.info("Calcul du découpage temporel Gold avec les quantiles chronologiques 70 pourcent et 85 pourcent.")
    q70 = hourly["window_start"].quantile(0.70)
    q85 = hourly["window_start"].quantile(0.85)
    df = hourly.copy()
    df["split_set"] = np.select(
        [df["window_start"] < q70, df["window_start"] < q85],
        ["train", "validation"],
        default="test",
    )
    LOGGER.info("Le découpage temporel Gold utilise q70=%s et q85=%s ; la répartition est %s.", q70, q85, df["split_set"].value_counts().to_dict())
    return df


def impute_hourly_signals_train_only(hourly: pd.DataFrame) -> pd.DataFrame:
    """Impute les signaux horaires par interpolation puis médianes ajustées sur le train."""

    df = hourly.copy()
    numeric_columns = list(SIGNAL_COLUMNS.values()) + [
        "temp_max_1h",
        "pressure_max_1h",
        "voltage_max_1h",
        "rotation_max_1h",
        "pieces_produced_1h",
    ]
    LOGGER.info("Imputation des signaux Silver pour les colonnes %s avec interpolation par machine puis médiane d'entraînement.", numeric_columns)
    train_medians = df.loc[df["split_set"] == "train", numeric_columns].median(numeric_only=True)
    missing_before = df[numeric_columns].isna().sum().sum()

    for column in numeric_columns:
        df[column] = df.groupby("machine_id_std", group_keys=False)[column].apply(lambda s: s.interpolate(limit_direction="both"))
        df[column] = df[column].fillna(train_medians[column])
    df["pieces_produced_1h"] = df["pieces_produced_1h"].clip(lower=0)

    missing_after = df[numeric_columns].isna().sum().sum()
    LOGGER.info("L'imputation Silver est terminée : %s valeurs manquantes avant, %s après. Médianes d'entraînement : %s.", f"{missing_before:,}", f"{missing_after:,}", train_medians.round(3).to_dict())
    return df


def add_machine_reference(hourly: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """Joint le contexte machine statique utilisé par les variables de production."""

    LOGGER.info("Ajout du référentiel machine Gold avec les capacités et les métadonnées.")
    ref = machines.rename(columns={"machine_code": "machine_id_std"})
    df = hourly.merge(ref, on="machine_id_std", how="left")
    missing = df["max_hourly_capacity_pieces"].isna().sum()
    if missing:
        LOGGER.warning("%s lignes Gold n'ont pas de référentiel machine associé.", f"{missing:,}")
    return df


def add_incident_context(hourly: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les incidents passés, leur sévérité, leurs types et le temps depuis le dernier incident."""

    LOGGER.info("Ajout du contexte incidents Gold avec agrégation horaire et historique sur 24 heures et 7 jours.")
    agg_map = {"incident_id": "count", "severity": "max", **{col: "sum" for col in TYPE_COLUMNS}}
    incident_hourly = incidents.groupby(["machine_id_std", "window_start"], as_index=False).agg(agg_map)
    incident_hourly = incident_hourly.rename(columns={"incident_id": "incident_count_1h", "severity": "incident_max_severity_1h"})

    df = hourly.merge(incident_hourly, on=["machine_id_std", "window_start"], how="left")
    fill_columns = ["incident_count_1h", "incident_max_severity_1h", *TYPE_COLUMNS]
    df[fill_columns] = df[fill_columns].fillna(0)

    grouped = df.groupby("machine_id_std", group_keys=False)
    df["incident_count_prev_24h"] = grouped["incident_count_1h"].apply(lambda s: s.rolling(24, min_periods=1).sum())
    df["incident_max_severity_prev_24h"] = grouped["incident_max_severity_1h"].apply(lambda s: s.rolling(24, min_periods=1).max())
    df["incident_count_prev_7d"] = grouped["incident_count_1h"].apply(lambda s: s.rolling(168, min_periods=1).sum())
    for column in TYPE_COLUMNS:
        df[f"{column}_count_prev_24h"] = grouped[column].apply(lambda s: s.rolling(24, min_periods=1).sum())

    df["hours_since_last_incident"] = _hours_since_last_event(df, "incident_count_1h")
    df["hours_since_last_incident"] = df["hours_since_last_incident"].fillna(-1)
    LOGGER.info("Le contexte incidents Gold rattache %s incidents au total et %s colonnes de types.", int(df["incident_count_1h"].sum()), len(TYPE_COLUMNS))
    return df


def add_maintenance_context(hourly: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables d'historique de maintenance sans utiliser d'interventions futures."""

    LOGGER.info("Ajout du contexte maintenance Gold avec le délai depuis la dernière maintenance et une fenêtre glissante de 30 jours.")
    maint_hourly = (
        maintenance.rename(columns={"machine_code": "machine_id_std", "maintenance_hour": "window_start"})
        .groupby(["machine_id_std", "window_start"], as_index=False)
        .agg(maintenance_count_1h=("maintenance_id", "count"))
    )
    df = hourly.merge(maint_hourly, on=["machine_id_std", "window_start"], how="left")
    df["maintenance_count_1h"] = df["maintenance_count_1h"].fillna(0)
    df["maintenance_count_prev_30d"] = df.groupby("machine_id_std", group_keys=False)["maintenance_count_1h"].apply(lambda s: s.rolling(720, min_periods=1).sum())
    df["days_since_last_maintenance"] = _hours_since_last_event(df, "maintenance_count_1h") / 24
    df["days_since_last_maintenance"] = df["days_since_last_maintenance"].fillna(-1)
    LOGGER.info("Le contexte maintenance Gold rattache %s maintenances.", int(df["maintenance_count_1h"].sum()))
    return df


def _hours_since_last_event(df: pd.DataFrame, event_count_column: str) -> pd.Series:
    """Renvoie le nombre d'heures écoulées depuis l'événement précédent par machine."""

    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby("machine_id_std", sort=False):
        last_event_time = pd.NaT
        values = []
        for _, row in group.iterrows():
            if row[event_count_column] > 0:
                last_event_time = row["window_start"]
                values.append(0.0)
            elif pd.isna(last_event_time):
                values.append(np.nan)
            else:
                values.append((row["window_start"] - last_event_time).total_seconds() / 3600)
        result.loc[group.index] = values
    return result


def add_memory_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Crée les moyennes, maximums et écarts-types glissants 6h, 12h et 24h par machine."""

    LOGGER.info("Calcul des fenêtres glissantes Gold sur 6 heures, 12 heures et 24 heures pour les signaux %s.", list(SIGNAL_COLUMNS))
    df = hourly.copy()
    grouped = df.groupby("machine_id_std", group_keys=False)
    for prefix, column in SIGNAL_COLUMNS.items():
        for window in (6, 12, 24):
            df[f"{prefix}_mean_{window}h"] = grouped[column].apply(lambda s, w=window: s.rolling(w, min_periods=1).mean())
            df[f"{prefix}_max_{window}h"] = grouped[column].apply(lambda s, w=window: s.rolling(w, min_periods=1).max())
            df[f"{prefix}_std_{window}h"] = grouped[column].apply(lambda s, w=window: s.rolling(w, min_periods=1).std()).fillna(0)
    LOGGER.info("Les fenêtres glissantes Gold ont créé %s colonnes.", 4 * 3 * 3)
    return df


def add_trend_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Crée la tendance 6h et les deltas 1h/3h par signal, groupés par machine."""

    LOGGER.info("Calcul des tendances Gold avec les deltas 1 heure, 3 heures et la tendance 6 heures par machine.")
    df = hourly.copy()
    grouped = df.groupby("machine_id_std", group_keys=False)
    for prefix, column in SIGNAL_COLUMNS.items():
        df[f"{prefix}_trend_6h"] = grouped[column].apply(lambda s: s - s.shift(6))
        df[f"{prefix}_delta_1h"] = grouped[column].apply(lambda s: s - s.shift(1))
        df[f"{prefix}_delta_3h"] = grouped[column].apply(lambda s: s - s.shift(3))
    trend_columns = [col for col in df.columns if col.endswith(("trend_6h", "delta_1h", "delta_3h"))]
    df[trend_columns] = df[trend_columns].fillna(0)
    return df


def add_anomaly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Crée les z-scores glissants et machine sans fuite validation/test."""

    LOGGER.info("Calcul des anomalies Gold avec z-score sur 24 heures et z-score machine ajusté uniquement sur l'entraînement.")
    df = hourly.copy()
    eps = 1e-9
    for prefix, column in SIGNAL_COLUMNS.items():
        mean_col = f"{prefix}_mean_24h"
        std_col = f"{prefix}_std_24h"
        df[f"{prefix}_zscore_24h"] = (df[column] - df[mean_col]) / df[std_col].replace(0, np.nan)
        train_stats = (
            df[df["split_set"] == "train"]
            .groupby("machine_id_std")[column]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{prefix}_train_mean", "std": f"{prefix}_train_std"})
        )
        df = df.merge(train_stats, on="machine_id_std", how="left")
        df[f"{prefix}_zscore_machine"] = (df[column] - df[f"{prefix}_train_mean"]) / df[f"{prefix}_train_std"].replace(0, np.nan)
        df = df.drop(columns=[f"{prefix}_train_mean", f"{prefix}_train_std"])

    zscore_columns = [column for column in df.columns if "zscore" in column]
    df[zscore_columns] = df[zscore_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    LOGGER.info("Les anomalies Gold ont créé %s colonnes z-score avec epsilon=%s.", len(zscore_columns), eps)
    return df


def add_production_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les indicateurs de charge de production prévus par la roadmap."""

    LOGGER.info("Calcul des variables de production Gold avec production glissante sur 24 heures et taux d'utilisation de capacité.")
    df = hourly.copy()
    df["pieces_produced_sum_24h"] = df.groupby("machine_id_std", group_keys=False)["pieces_produced_1h"].apply(lambda s: s.rolling(24, min_periods=1).sum())
    df["capacity_utilization_pct"] = np.where(
        df["max_hourly_capacity_pieces"].fillna(0) > 0,
        100 * df["pieces_produced_1h"] / df["max_hourly_capacity_pieces"],
        np.nan,
    )
    df["capacity_utilization_pct"] = df["capacity_utilization_pct"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def add_future_labels(hourly: pd.DataFrame) -> pd.DataFrame:
    """Crée les labels de panne future avec une anticipation glissante inversée par machine."""

    LOGGER.info("Création des étiquettes Gold pour les horizons 6 heures, 12 heures, 24 heures et 48 heures.")
    df = hourly.copy()
    grouped = df.groupby("machine_id_std", group_keys=False)
    for horizon in (6, 12, 24, 48):
        future_col = f"future_incident_count_{horizon}h"
        label_col = f"label_failure_next_{horizon}h"
        df[future_col] = grouped["incident_count_1h"].apply(
            lambda s, h=horizon: s.iloc[::-1].rolling(h, min_periods=1).sum().iloc[::-1]
        )
        df[label_col] = df[future_col] > 0
        LOGGER.info("L'étiquette %s contient %s exemples positifs.", label_col, int(df[label_col].sum()))
    df["window_end"] = df["window_start"] + pd.Timedelta(hours=1)
    return df


def order_gold_columns(gold: pd.DataFrame) -> pd.DataFrame:
    """Place les identifiants au début, les étiquettes à la fin et regroupe les variables."""

    first = ["machine_id_std", "window_start", "window_end", "split_set"]
    labels = [f"label_failure_next_{h}h" for h in (6, 12, 24, 48)] + [f"future_incident_count_{h}h" for h in (6, 12, 24, 48)]
    remaining = [column for column in gold.columns if column not in first + labels]
    ordered = first + remaining + labels
    return gold[ordered].sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)


def validate_gold_dataset(gold: pd.DataFrame) -> None:
    """Exécute les contrôles contractuels issus des règles non négociables du document."""

    LOGGER.info("Validation du Gold Dataset : unicité machine-heure, découpages temporels, étiquettes et volume de colonnes.")
    duplicates = gold.duplicated(["machine_id_std", "window_start"]).sum()
    if duplicates:
        raise ValueError(f"Gold Dataset invalide: {duplicates} doublons machine+heure")

    if gold[["machine_id_std", "window_start", "window_end", "split_set"]].isna().any().any():
        raise ValueError("Gold Dataset invalide: identifiants incomplets")

    split_order = gold.groupby("split_set")["window_start"].agg(["min", "max"]).to_dict("index")
    LOGGER.info("Les bornes temporelles des découpages Gold sont %s.", split_order)
    if gold.shape[1] < 90:
        LOGGER.warning("Le Gold Dataset contient %s colonnes, ce qui est inférieur à l'objectif indicatif de 100 colonnes.", gold.shape[1])
    LOGGER.info("La validation Gold est terminée : %s colonnes et %s valeurs manquantes au total.", gold.shape[1], f"{gold.isna().sum().sum():,}")


def persist_gold_dataset(gold: pd.DataFrame, config: GoldDatasetConfig, engine=None) -> Path:
    """Persiste le Gold Dataset large en CSV et, si demandé, en table PostgreSQL."""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{timestamp}_gold_dataset"
    run_dir = config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"gold_dataset_{timestamp}.csv"
    gold.to_csv(csv_path, index=False)
    LOGGER.info("Le Gold Dataset a été enregistré en CSV dans %s avec %s lignes et %s colonnes.", csv_path, f"{len(gold):,}", gold.shape[1])

    database_url = config.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    if config.persist_db and database_url:
        if config.auto_start_docker and engine is None:
            ensure_postgres_stack_running()
        target_engine = engine or create_engine(database_url, future=True)
        write_gold_table_wide(gold, target_engine, config.table_name)
    write_gold_run_metadata(gold, run_name, run_dir, csv_path, config)
    return csv_path


def write_gold_table_wide(gold: pd.DataFrame, engine, table_name: str) -> None:
    """Écrit la table Gold avec toutes les colonnes du DataFrame final."""

    data = gold.copy()
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        connection.execute(text(f'DROP TABLE IF EXISTS "gold"."{table_name}" CASCADE'))
    data.to_sql(table_name, engine, schema="gold", if_exists="replace", index=False, chunksize=1000)
    LOGGER.info(
        "Écriture du Gold Dataset large dans la table gold.%s : %s lignes et %s colonnes.",
        table_name,
        f"{len(data):,}",
        data.shape[1],
    )


def write_gold_run_metadata(
    gold: pd.DataFrame,
    run_name: str,
    run_dir: Path,
    csv_path: Path,
    config: GoldDatasetConfig,
) -> None:
    """Enregistre les artefacts Gold dans le meme index que les autres runs."""

    report_path = run_dir / "rapport_gold_dataset.md"
    metadata_path = run_dir / "metadata.json"
    split_counts = {str(key): int(value) for key, value in gold["split_set"].value_counts().items()} if "split_set" in gold else {}
    metadata = {
        "run_ts": run_name.split("_", 1)[0],
        "run_name": run_name,
        "layer": "gold",
        "source_layer": "silver" if config.persist_db else "files",
        "schema": "gold",
        "run_dir": project_relative(run_dir),
        "report_path": project_relative(report_path),
        "gold_csv_path": project_relative(csv_path),
        "nombre_graphes": 0,
        "nombre_lignes": int(len(gold)),
        "nombre_lignes_telemetrie_lues": 0,
        "nombre_lignes_telemetrie_utilisees": 0,
        "machines_uniques": int(gold["machine_id_std"].nunique()) if "machine_id_std" in gold else 0,
        "nombre_colonnes": int(gold.shape[1]),
        "split_counts": split_counts,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Gold Dataset",
                "",
                f"- Run : `{run_name}`",
                f"- CSV : `{project_relative(csv_path)}`",
                f"- Lignes : {len(gold):,}",
                f"- Colonnes : {gold.shape[1]:,}",
                f"- Machines : {metadata['machines_uniques']:,}",
                f"- Splits : `{split_counts}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    update_incident_run_indexes(metadata)
    LOGGER.info("Les artefacts du run Gold ont été enregistrés dans le dossier %s.", run_dir)


def to_gold_storage_frame(gold: pd.DataFrame) -> pd.DataFrame:
    """Transforme le dataset large en table Gold stable: identités, features JSON, labels JSON."""

    identity_columns = ["machine_id_std", "window_start", "window_end", "split_set"]
    label_columns = [f"label_failure_next_{h}h" for h in (6, 12, 24, 48)] + [
        f"future_incident_count_{h}h" for h in (6, 12, 24, 48)
    ]
    feature_columns = [column for column in gold.columns if column not in identity_columns + label_columns]

    rows = []
    for record in gold.to_dict(orient="records"):
        rows.append(
            {
                **{column: record[column] for column in identity_columns},
                "features": {column: json_safe_value(record[column]) for column in feature_columns},
                "labels": {column: json_safe_value(record[column]) for column in label_columns},
            }
        )
    return pd.DataFrame(rows)


def expand_gold_storage_frame(stored: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit le DataFrame large depuis la représentation Gold JSONB."""

    rows = []
    for record in stored.to_dict(orient="records"):
        features = record.get("features") or {}
        labels = record.get("labels") or {}
        if isinstance(features, str):
            features = json.loads(features)
        if isinstance(labels, str):
            labels = json.loads(labels)
        rows.append(
            {
                "machine_id_std": record["machine_id_std"],
                "window_start": record["window_start"],
                "window_end": record["window_end"],
                "split_set": record["split_set"],
                **features,
                **labels,
            }
        )
    return order_gold_columns(pd.DataFrame(rows))


def json_safe_value(value):
    """Convertit les scalaires pandas/numpy en valeurs compatibles JSONB."""

    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def ensure_postgres_stack_running() -> None:
    """Démarre le stack PostgreSQL Docker Compose du projet si nécessaire.

    LOGGER.info("Démarrage de la couche Bronze : lecture des sources et chargement dans PostgreSQL.")
    PostgreSQL vit dans le conteneur ``formation-postgres`` et pgAdmin est
    exposé sur le port 5050 pour inspection visuelle.
    """

    project_dir = find_project_dir(Path.cwd())
    compose_file = project_dir / "docker-compose.yml"
    if not compose_file.exists():
        raise FileNotFoundError("docker-compose.yml est introuvable à la racine du projet.")

    docker_command = find_docker_command()
    LOGGER.info("Vérification du moteur Docker avec la commande %s.", docker_command)
    if not docker_is_ready(docker_command, project_dir):
        start_docker_desktop()
        wait_for_docker(docker_command, project_dir)

    if docker_container_exists(docker_command, project_dir, POSTGRES_CONTAINER):
        LOGGER.info("Le conteneur PostgreSQL existe déjà. Il va être réutilisé au lieu d'être recréé.")
        start_existing_container(docker_command, project_dir, POSTGRES_CONTAINER)
        if docker_container_exists(docker_command, project_dir, PGADMIN_CONTAINER):
            start_existing_container(docker_command, project_dir, PGADMIN_CONTAINER)
        wait_for_postgres(docker_command, project_dir)
        LOGGER.info("PostgreSQL est prêt. pgAdmin est disponible sur %s avec les identifiants admin@example.com/admin.", PGADMIN_URL)
        return

    LOGGER.info("Démarrage de PostgreSQL et pgAdmin avec Docker Compose.")
    compose_up = run_command(
        [docker_command, "compose", "up", "-d", "postgres", "pgadmin"],
        cwd=project_dir,
        check=False,
    )
    if compose_up.returncode != 0:
        error_output = f"{compose_up.stdout}\n{compose_up.stderr}"
        if "already in use" in error_output and docker_container_exists(docker_command, project_dir, POSTGRES_CONTAINER):
            LOGGER.warning(
                "Docker Compose signale que le conteneur PostgreSQL existe déjà. "
                "Le backend réutilise ce conteneur existant."
            )
            start_existing_container(docker_command, project_dir, POSTGRES_CONTAINER)
            wait_for_postgres(docker_command, project_dir)
            LOGGER.info("PostgreSQL est prêt. pgAdmin est disponible sur %s avec les identifiants admin@example.com/admin.", PGADMIN_URL)
            return
        raise RuntimeError(f"Impossible de démarrer PostgreSQL/pgAdmin : {compose_up.stderr.strip()}")

    wait_for_postgres(docker_command, project_dir)
    LOGGER.info("PostgreSQL est prêt. pgAdmin est disponible sur %s avec les identifiants admin@example.com/admin.", PGADMIN_URL)


def find_project_dir(start: Path) -> Path:
    """Trouve la racine du projet contenant ``data`` et ``docker-compose.yml``."""

    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "docker-compose.yml").exists():
            return candidate
    raise FileNotFoundError("Impossible de trouver la racine du projet contenant data/ et docker-compose.yml.")


def run_command(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Exécute un sous-processus en conservant stdout/stderr pour les logs et erreurs."""

    try:
        return subprocess.run(command, check=check, text=True, capture_output=True, cwd=cwd)
    except FileNotFoundError as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def find_docker_command() -> str:
    """Localise le CLI Docker depuis le PATH ou les chemins Windows habituels."""

    docker_cli = shutil.which("docker")
    if docker_cli:
        return docker_cli

    candidates = [
        Path(os.environ.get("ProgramFiles", "")) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Docker" / "Docker" / "resources" / "docker.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "docker"


def docker_is_ready(docker_command: str, project_dir: Path) -> bool:
    """Renvoie True quand le moteur Docker répond à ``docker version``."""

    return run_command([docker_command, "version"], cwd=project_dir, check=False).returncode == 0


def docker_container_exists(docker_command: str, project_dir: Path, container_name: str) -> bool:
    """Renvoie True si un conteneur Docker portant ce nom existe déjà."""

    result = run_command(
        [docker_command, "container", "inspect", container_name],
        cwd=project_dir,
        check=False,
    )
    return result.returncode == 0


def start_existing_container(docker_command: str, project_dir: Path, container_name: str) -> None:
    """Démarre un conteneur existant, ou ne fait rien s'il est déjà lancé."""

    result = run_command(
        [docker_command, "start", container_name],
        cwd=project_dir,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode == 0:
        LOGGER.info("Le conteneur Docker %s est démarré.", container_name)
        return
    if "already started" in output or "is already running" in output:
        LOGGER.info("Le conteneur Docker %s est déjà démarré.", container_name)
        return
    raise RuntimeError(f"Impossible de démarrer le conteneur Docker {container_name} : {result.stderr.strip()}")


def start_docker_desktop() -> None:
    """Démarre Docker Desktop sous Windows quand le moteur Docker n'est pas prêt."""

    candidates = [
        Path(os.environ.get("ProgramFiles", "")) / "Docker" / "Docker" / "Docker Desktop.exe",
        Path(os.environ.get("LocalAppData", "")) / "Docker" / "Docker Desktop.exe",
    ]
    docker_desktop = next((path for path in candidates if path.exists()), None)
    if docker_desktop is None:
        raise RuntimeError("Docker Desktop est introuvable. Lance Docker Desktop puis relance le pipeline.")

    LOGGER.info("Le moteur Docker n'est pas prêt ; lancement de Docker Desktop.")
    subprocess.Popen([str(docker_desktop)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_docker(docker_command: str, project_dir: Path, attempts: int = 60) -> None:
    """Attend jusqu'à deux minutes que Docker Desktop expose son moteur."""

    for attempt in range(1, attempts + 1):
        if docker_is_ready(docker_command, project_dir):
            LOGGER.info("Le moteur Docker est prêt après %s tentative(s).", attempt)
            return
        LOGGER.info("En attente du moteur Docker, tentative %s sur %s.", attempt, attempts)
        time.sleep(2)
    raise RuntimeError("Docker Desktop n'est pas prêt après 120 secondes.")


def wait_for_postgres(docker_command: str, project_dir: Path, attempts: int = 30) -> None:
    """Attend que le conteneur PostgreSQL accepte les connexions."""

    for attempt in range(1, attempts + 1):
        ready = run_command(
            [docker_command, "exec", POSTGRES_CONTAINER, "pg_isready", "-U", POSTGRES_USER, "-d", POSTGRES_DB],
            cwd=project_dir,
            check=False,
        )
        if ready.returncode == 0:
            LOGGER.info("PostgreSQL accepte les connexions après %s tentative(s).", attempt)
            return
        LOGGER.info("En attente de PostgreSQL, tentative %s sur %s.", attempt, attempts)
        time.sleep(2)
    raise RuntimeError("PostgreSQL n'est pas prêt après 60 secondes.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Analyse les options CLI pour des exécutions locales reproductibles."""

    parser = argparse.ArgumentParser(description="Construit les couches Bronze, Silver et Gold InduSense 4.0.")
    parser.add_argument("--telemetry", type=Path, default=Path("data/telemetry.csv"))
    parser.add_argument("--incidents", type=Path, default=Path("data/releves_incidents.csv"))
    parser.add_argument("--machine-sql", type=Path, default=Path("data/machine.sql"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gold-datasets"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL)
    parser.add_argument(
        "--layer",
        "--stage",
        choices=["bronze", "silver", "gold", "all"],
        default="all",
        help="Couche à exécuter. Par défaut: all = bronze puis silver puis gold.",
    )
    parser.add_argument("--no-db", action="store_true", help="Ne pas tenter la persistance PostgreSQL.")
    parser.add_argument("--no-docker", action="store_true", help="Ne pas démarrer Docker Compose automatiquement.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Point d'entrée CLI utilisé par ``python -m indusense.processing.ingestion``."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    config = GoldDatasetConfig(
        telemetry_path=args.telemetry,
        incidents_path=args.incidents,
        machine_sql_path=args.machine_sql,
        output_dir=args.output_dir,
        persist_db=not args.no_db,
        database_url=args.database_url,
        auto_start_docker=not args.no_docker,
        layer=args.layer,
    )
    result = run_layer_pipeline(config)
    if result is not None:
        LOGGER.info("Aperçu de la répartition split_set : %s.", result["split_set"].value_counts().to_dict())


if __name__ == "__main__":
    main()
