"""Mesure reproductible de l'empreinte carbone d'un entraînement vision."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from indusense.config import COUNTRY_ISO_CODE


class VisionEmissionsTracker(AbstractContextManager["VisionEmissionsTracker"]):
    """Encapsule CodeCarbon et reste utilisable lorsque le paquet est absent.

    Le mode dégradé ne fabrique aucune estimation : il enregistre explicitement
    que la mesure doit être activée via l'extra ``dl``.
    """

    def __init__(self, output_dir: Path, country_iso_code: str = COUNTRY_ISO_CODE) -> None:
        self.output_dir = output_dir
        self.country_iso_code = country_iso_code
        self._tracker: Any | None = None
        self._started_at = 0.0
        self.result: dict[str, Any] = {"available": False, "duration_seconds": 0.0, "energy_kwh": None, "emissions_gco2eq": None}

    def __enter__(self) -> "VisionEmissionsTracker":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.perf_counter()
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(
                output_dir=str(self.output_dir),
                output_file="emissions.csv",
                country_iso_code=self.country_iso_code,
                save_to_file=True,
                log_level="error",
            )
            self._tracker.start()
            self.result["available"] = True
        except ImportError:
            self.result["reason"] = "codecarbon non installé"
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.result["duration_seconds"] = round(time.perf_counter() - self._started_at, 3)
        if self._tracker is None:
            return None
        emissions_kg = self._tracker.stop()
        self.result["emissions_gco2eq"] = None if emissions_kg is None else float(emissions_kg) * 1000
        # CodeCarbon expose l'énergie cumulée sur le tracker selon sa version.
        energy = getattr(getattr(self._tracker, "_total_energy", None), "kWh", None)
        self.result["energy_kwh"] = None if energy is None else float(energy)
        self.result["emissions_csv"] = "emissions.csv"
        return None
