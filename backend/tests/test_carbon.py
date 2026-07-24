from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from indusense.vision.carbon import VisionEmissionsTracker


class _Energy:
    kWh = 0.001


class _OfflineTracker:
    kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        self.__class__.kwargs = kwargs
        self._total_energy = _Energy()

    def start(self) -> None:
        return None

    def stop(self) -> float:
        return 0.0002


class CarbonTrackerTests(unittest.TestCase):
    def test_uses_offline_tracker_when_country_is_explicit(self) -> None:
        fake_codecarbon = types.SimpleNamespace(OfflineEmissionsTracker=_OfflineTracker)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(sys.modules, {"codecarbon": fake_codecarbon}):
                with VisionEmissionsTracker(Path(temporary_directory)) as tracker:
                    pass

        self.assertEqual(_OfflineTracker.kwargs["country_iso_code"], "FRA")
        self.assertTrue(tracker.result["available"])
        self.assertEqual(tracker.result["energy_kwh"], 0.001)
        self.assertEqual(tracker.result["emissions_gco2eq"], 0.2)

    def test_codecarbon_failure_does_not_abort_training(self) -> None:
        class BrokenTracker:
            def __init__(self, **_: object) -> None:
                raise TypeError("configuration incompatible")

        fake_codecarbon = types.SimpleNamespace(OfflineEmissionsTracker=BrokenTracker)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(sys.modules, {"codecarbon": fake_codecarbon}):
                with VisionEmissionsTracker(Path(temporary_directory)) as tracker:
                    pass

        self.assertFalse(tracker.result["available"])
        self.assertIn("configuration incompatible", tracker.result["reason"])


if __name__ == "__main__":
    unittest.main()
