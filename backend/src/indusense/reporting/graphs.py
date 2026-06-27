"""Generation Python des graphes Bronze/Silver."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text

from indusense.artifacts import (
    INCIDENT_ARTIFACT_ROOT,
    PROJECT_DIR,
    project_relative,
    update_incident_run_indexes,
)
from indusense.processing.ingestion import DEFAULT_DATABASE_URL, TYPE_COLUMNS


LOGGER = logging.getLogger(__name__)
ARTIFACT_ROOT = INCIDENT_ARTIFACT_ROOT


@dataclass(frozen=True)
class GraphReportResult:
    run_name: str
    run_dir: Path
    graphs_dir: Path
    report_path: Path
    metadata_path: Path
    graph_count: int
    incident_rows: int
    telemetry_rows: int
    machines: int


def generate_graph_report(source_layer: str, database_url: str | None = None) -> GraphReportResult:
    """Genere les graphes d'analyse pour une couche Bronze ou Silver."""

    if source_layer not in {"bronze", "silver"}:
        raise ValueError("source_layer doit valoir bronze ou silver")

    sns.set_theme(style="whitegrid", context="notebook")
    database_url = database_url or DEFAULT_DATABASE_URL
    engine = create_engine(database_url)
    run_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{run_ts}_{source_layer}_report"
    run_dir = ARTIFACT_ROOT / run_name
    graph_dir = run_dir / "graphs"
    graph_dirs = {
        "temps": graph_dir / "01_temps",
        "severite": graph_dir / "02_severite",
        "commentaires": graph_dir / "03_commentaires",
        "telemetrie": graph_dir / "04_telemetrie",
        "machines_maintenance": graph_dir / "05_machines_maintenance",
        "qualite": graph_dir / "06_qualite_donnees",
    }
    for directory in graph_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Lecture des données de la couche « %s » depuis PostgreSQL pour générer les graphes.", source_layer)
    telemetry, incidents, machines, maintenance = load_layer_frames(engine, source_layer)
    graph_paths: list[Path] = []

    graph_paths.extend(generate_time_graphs(incidents, graph_dirs["temps"]))
    graph_paths.extend(generate_severity_graphs(incidents, graph_dirs["severite"]))
    graph_paths.extend(generate_comment_graphs(incidents, graph_dirs["commentaires"]))
    graph_paths.extend(generate_telemetry_graphs(telemetry, incidents, graph_dirs["telemetrie"]))
    graph_paths.extend(generate_machine_maintenance_graphs(incidents, machines, maintenance, graph_dirs["machines_maintenance"]))
    graph_paths.extend(generate_quality_graphs(telemetry, incidents, graph_dirs["qualite"]))

    report_path = run_dir / f"rapport_graphes_{source_layer}.md"
    metadata_path = run_dir / "metadata.json"
    metadata = {
        "run_ts": run_ts,
        "run_name": run_name,
        "layer": "report",
        "source_layer": source_layer,
        "schema": source_layer,
        "telemetry_table": "telemetry_raw" if source_layer == "bronze" else "telemetry",
        "run_dir": project_relative(run_dir),
        "graphs_dir": project_relative(graph_dir),
        "report_path": project_relative(report_path),
        "nombre_graphes": len(graph_paths),
        "nombre_lignes": int(len(incidents)),
        "nombre_lignes_telemetrie_lues": int(len(telemetry)),
        "nombre_lignes_telemetrie_utilisees": int(len(telemetry.dropna(subset=["timestamp"]))),
        "machines_uniques": int(telemetry["machine_id"].nunique()) if "machine_id" in telemetry else 0,
    }
    write_report(report_path, source_layer, metadata, graph_paths)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    update_incident_run_indexes(metadata)
    LOGGER.info("Le rapport de graphes « %s » a été généré avec %s graphes.", run_name, len(graph_paths))

    return GraphReportResult(
        run_name=run_name,
        run_dir=run_dir,
        graphs_dir=graph_dir,
        report_path=report_path,
        metadata_path=metadata_path,
        graph_count=len(graph_paths),
        incident_rows=len(incidents),
        telemetry_rows=len(telemetry),
        machines=metadata["machines_uniques"],
    )


def load_layer_frames(engine, source_layer: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    telemetry_table = "telemetry_raw" if source_layer == "bronze" else "telemetry"
    incidents_table = "incidents_raw" if source_layer == "bronze" else "incidents"
    telemetry = read_table(engine, source_layer, telemetry_table)
    incidents = read_table(engine, source_layer, incidents_table)
    machines = read_table(engine, source_layer, "machine")
    maintenance = read_table(engine, source_layer, "maintenance")

    telemetry = normalize_machine_column(telemetry)
    incidents = normalize_machine_column(incidents)
    maintenance = normalize_machine_column(maintenance)
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], errors="coerce")
    incidents["incident_at"] = build_incident_timestamp(incidents)
    maintenance["maintenance_at"] = pd.to_datetime(maintenance["maintenance_at"], errors="coerce")
    return telemetry, incidents, machines, maintenance


def normalize_machine_column(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "machine_id_std" in normalized:
        normalized["machine_id"] = normalized["machine_id_std"]
    elif "machine_code" in normalized:
        normalized["machine_id"] = normalized["machine_code"]
    return normalized


def read_table(engine, schema: str, table: str) -> pd.DataFrame:
    return pd.read_sql_query(text(f'SELECT * FROM "{schema}"."{table}"'), engine)


def build_incident_timestamp(incidents: pd.DataFrame) -> pd.Series:
    if "incident_at" in incidents:
        return pd.to_datetime(incidents["incident_at"], errors="coerce")
    return pd.to_datetime(incidents["date"].astype(str) + " " + incidents["time"].astype(str), errors="coerce")


def generate_time_graphs(incidents: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    data = incidents.dropna(subset=["incident_at"]).copy()
    if data.empty:
        return [empty_plot(output_dir / "distribution_incidents_par_jour.png", "Distribution incidents par jour")]

    daily = data.set_index("incident_at").resample("D").size().rename("incidents").reset_index()
    paths.append(save_lineplot(daily, "incident_at", "incidents", output_dir / "distribution_incidents_par_jour.png", "Incidents par jour"))

    data["hour"] = data["incident_at"].dt.hour
    data["weekday"] = data["incident_at"].dt.day_name()
    pivot = data.pivot_table(index="weekday", columns="hour", values="incident_id", aggfunc="count", fill_value=0)
    paths.append(save_heatmap(pivot, output_dir / "heure_journee_incidents_heatmap.png", "Incidents par jour de semaine et heure"))

    if "shift" in data:
        paths.append(save_countplot(data, "shift", output_dir / "distribution_incidents_par_shift.png", "Incidents par shift"))
    return paths


def generate_severity_graphs(incidents: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    if "severity" not in incidents:
        return [empty_plot(output_dir / "repartition_incidents_par_severite.png", "Repartition severite")]
    paths.append(save_countplot(incidents, "severity", output_dir / "repartition_incidents_par_severite.png", "Repartition des incidents par severite"))
    top = incidents.groupby("machine_id", as_index=False)["severity"].agg(["count", "mean"]).reset_index()
    if not top.empty:
        top = top.sort_values("count", ascending=False).head(15)
        paths.append(save_barplot(top, "machine_id", "count", output_dir / "top_machines_incidents.png", "Top machines par nombre d'incidents"))
    type_cols = [column for column in TYPE_COLUMNS if column in incidents]
    if type_cols:
        matrix = incidents.groupby("severity")[type_cols].sum()
        paths.append(save_heatmap(matrix, output_dir / "severite_par_type_incident_heatmap.png", "Severite par type d'incident"))
    return paths


def generate_comment_graphs(incidents: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    if "comment" not in incidents:
        return []
    comments = incidents["comment"].fillna("").astype(str).str.strip()
    completeness = pd.DataFrame({"statut": comments.ne("").map({True: "commentaire", False: "vide"})})
    paths.append(save_countplot(completeness, "statut", output_dir / "commentaires_presence.png", "Presence des commentaires"))
    frequent = comments[comments.ne("")].value_counts().head(20).rename_axis("comment").reset_index(name="nombre")
    if not frequent.empty:
        paths.append(save_barplot(frequent, "comment", "nombre", output_dir / "commentaires_frequents.png", "Commentaires frequents", rotate=True))
    return paths


def generate_telemetry_graphs(telemetry: pd.DataFrame, incidents: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    signal_cols = ["temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm", "pieces_produced"]
    for signal in [column for column in signal_cols if column in telemetry]:
        sample = telemetry[["machine_id", signal]].dropna()
        if not sample.empty:
            paths.append(save_boxplot(sample, "machine_id", signal, output_dir / f"{signal}_par_machine.png", f"{signal} par machine"))

    numeric = telemetry[[column for column in signal_cols if column in telemetry]].dropna()
    if numeric.shape[1] >= 2:
        paths.append(save_heatmap(numeric.corr(), output_dir / "correlation_signaux_telemetrie.png", "Correlation des signaux telemetrie"))

    if "timestamp" in telemetry and not incidents.empty:
        hourly_incidents = incidents.assign(timestamp=incidents["incident_at"].dt.floor("h")).groupby(["machine_id", "timestamp"]).size().rename("incidents").reset_index()
        merged = telemetry.merge(hourly_incidents, on=["machine_id", "timestamp"], how="left")
        merged["incidents"] = merged["incidents"].fillna(0)
        corr_cols = [column for column in signal_cols if column in merged] + ["incidents"]
        corr = merged[corr_cols].corr(numeric_only=True)
        if "incidents" in corr:
            paths.append(save_heatmap(corr, output_dir / "correlation_incidents_signaux.png", "Correlation incidents / signaux"))
    return paths


def generate_machine_maintenance_graphs(incidents: pd.DataFrame, machines: pd.DataFrame, maintenance: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    incident_counts = incidents.groupby("machine_id").size().rename("incidents")
    maintenance_counts = maintenance.groupby("machine_id").size().rename("maintenances")
    volume = pd.concat([incident_counts, maintenance_counts], axis=1).fillna(0).reset_index()
    if not volume.empty:
        melted = volume.melt(id_vars="machine_id", var_name="indicateur", value_name="nombre")
        paths.append(save_grouped_barplot(melted, "machine_id", "nombre", "indicateur", output_dir / "machines_incidents_maintenances.png", "Incidents et maintenances par machine"))

    if "component" in maintenance and not maintenance.empty:
        matrix = maintenance.pivot_table(index="machine_id", columns="component", values="maintenance_id", aggfunc="count", fill_value=0)
        paths.append(save_heatmap(matrix, output_dir / "maintenances_machine_composant.png", "Maintenances par machine et composant"))

    if "criticality" in machines:
        paths.append(save_countplot(machines, "criticality", output_dir / "machines_par_criticite.png", "Machines par criticite"))
    return paths


def generate_quality_graphs(telemetry: pd.DataFrame, incidents: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    missing = telemetry.isna().mean().sort_values(ascending=False).head(20).rename_axis("colonne").reset_index(name="part_manquante")
    paths.append(save_barplot(missing, "colonne", "part_manquante", output_dir / "telemetrie_donnees_absentes.png", "Part de donnees manquantes telemetrie", rotate=True))

    if {"machine_id", "timestamp"}.issubset(telemetry.columns):
        duplicates = telemetry.duplicated(["machine_id", "timestamp"]).sum()
        dup_df = pd.DataFrame({"controle": ["doublons machine+timestamp"], "nombre": [duplicates]})
        paths.append(save_barplot(dup_df, "controle", "nombre", output_dir / "telemetrie_doublons.png", "Doublons telemetrie"))

    incident_missing = incidents.isna().mean().sort_values(ascending=False).head(20).rename_axis("colonne").reset_index(name="part_manquante")
    paths.append(save_barplot(incident_missing, "colonne", "part_manquante", output_dir / "incidents_donnees_absentes.png", "Part de donnees manquantes incidents", rotate=True))
    return paths


def save_lineplot(data: pd.DataFrame, x: str, y: str, path: Path, title: str) -> Path:
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=data, x=x, y=y, marker="o")
    return finish_plot(path, title)


def save_countplot(data: pd.DataFrame, x: str, path: Path, title: str) -> Path:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x=x, order=sorted(data[x].dropna().unique()))
    plt.xticks(rotation=30, ha="right")
    return finish_plot(path, title)


def save_barplot(data: pd.DataFrame, x: str, y: str, path: Path, title: str, rotate: bool = False) -> Path:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x=x, y=y)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    return finish_plot(path, title)


def save_grouped_barplot(data: pd.DataFrame, x: str, y: str, hue: str, path: Path, title: str) -> Path:
    plt.figure(figsize=(13, 6))
    sns.barplot(data=data, x=x, y=y, hue=hue)
    plt.xticks(rotation=45, ha="right")
    return finish_plot(path, title)


def save_boxplot(data: pd.DataFrame, x: str, y: str, path: Path, title: str) -> Path:
    plt.figure(figsize=(13, 6))
    sns.boxplot(data=data, x=x, y=y)
    plt.xticks(rotation=45, ha="right")
    return finish_plot(path, title)


def save_heatmap(data: pd.DataFrame, path: Path, title: str) -> Path:
    plt.figure(figsize=(12, max(5, 0.35 * len(data))))
    sns.heatmap(data, annot=True, fmt=".2g", cmap="YlGnBu", linewidths=0.5)
    return finish_plot(path, title)


def empty_plot(path: Path, title: str) -> Path:
    plt.figure(figsize=(10, 4))
    plt.text(0.5, 0.5, "Aucune donnee exploitable", ha="center", va="center", fontsize=14)
    plt.axis("off")
    return finish_plot(path, title)


def finish_plot(path: Path, title: str) -> Path:
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    LOGGER.info("Le graphe a été enregistré dans le fichier %s.", path)
    return path


def write_report(report_path: Path, source_layer: str, metadata: dict, graph_paths: list[Path]) -> None:
    lines = [
        f"# Rapport graphes {source_layer}",
        "",
        f"- Run : `{metadata['run_name']}`",
        f"- Incidents : {metadata['nombre_lignes']:,}",
        f"- Telemetrie : {metadata['nombre_lignes_telemetrie_lues']:,}",
        f"- Machines : {metadata['machines_uniques']:,}",
        f"- Graphes : {metadata['nombre_graphes']:,}",
        "",
        "## Graphes",
        "",
    ]
    for path in graph_paths:
        lines.append(f"- [{path.name}]({path.relative_to(report_path.parent).as_posix()})")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
