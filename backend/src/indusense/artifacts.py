"""Utilitaires communs pour indexer les artefacts d'execution."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
INCIDENT_ARTIFACT_ROOT = PROJECT_DIR / "artifacts" / "ingestions" / "incidents"
RUN_INDEX_JSON = INCIDENT_ARTIFACT_ROOT / "runs.json"
RUN_INDEX_MD = INCIDENT_ARTIFACT_ROOT / "runs.md"


def project_relative(path: Path) -> str:
    """Retourne un chemin relatif projet quand c'est possible."""

    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def update_incident_run_indexes(metadata: dict) -> None:
    """Ajoute ou remplace un run dans les index historiques incidents."""

    INCIDENT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = []
    if RUN_INDEX_JSON.exists():
        runs = json.loads(RUN_INDEX_JSON.read_text(encoding="utf-8"))

    run_key = metadata.get("run_name") or metadata.get("run_id") or metadata.get("run_ts")
    runs = [
        run
        for run in runs
        if (run.get("run_name") or run.get("run_id") or run.get("run_ts")) != run_key
    ]
    runs.append(metadata)
    RUN_INDEX_JSON.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Runs d'analyse incidents",
        "",
        "| Run | Type | Source | Schema | Incidents | Telemetrie | Graphes | Dossier |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for run in runs:
        run_name = run.get("run_name") or run.get("run_id") or run.get("run_ts")
        lines.append(
            f"| {run_name} | {run.get('layer')} | {run.get('source_layer', '')} | "
            f"{run.get('schema', '')} | {run.get('nombre_lignes', 0)} | "
            f"{run.get('nombre_lignes_telemetrie_lues', 0)} | {run.get('nombre_graphes', 0)} | "
            f"`{run.get('run_dir', '')}` |"
        )
    RUN_INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
