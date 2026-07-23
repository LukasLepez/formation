"""Accès au dataset MVTec de la pipeline vision.

Le module historique ``vision_dataset`` reste importable pour ne pas casser
l'API existante ; les nouveaux appels passent par ce point d'entrée B7.
"""

from indusense.vision_dataset import (
    VisionPreparationConfig,
    load_latest_preparation,
    load_model_input,
    prepare_vision_dataset,
)

__all__ = [
    "VisionPreparationConfig",
    "load_latest_preparation",
    "load_model_input",
    "prepare_vision_dataset",
]
