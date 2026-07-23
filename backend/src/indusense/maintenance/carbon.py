"""Instrumentation CodeCarbon des études ML."""

from indusense.vision.carbon import VisionEmissionsTracker

MaintenanceEmissionsTracker = VisionEmissionsTracker

__all__ = ["MaintenanceEmissionsTracker"]
