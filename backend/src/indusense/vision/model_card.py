"""Model card Hugging Face des modèles de détection d'anomalies visuelles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MODEL_CARD_VERSION = "1.0.0"


def generate_vision_model_card(report: dict[str, Any], run_dir: Path) -> Path:
    """Génère une fiche complète depuis le rapport réel d'un run vision."""

    model_type = report.get("model_type", "autoencoder")
    is_patchcore = model_type == "patchcore"
    model_label = "PatchCore avec ResNet-18" if is_patchcore else "auto-encodeur convolutionnel"
    image = (report.get("metrics") or {}).get("image") or {}
    pixel = (report.get("metrics") or {}).get("pixel") or {}
    confusion = image.get("confusion_matrix") or {}
    threshold = report.get("threshold") or {}
    training = report.get("training") or {}
    architecture = report.get("architecture") or {}
    config = report.get("config") or {}
    carbon = report.get("carbon") or {}
    reproducibility = report.get("reproducibility") or {}
    author = os.getenv("INDUSENSE_MODEL_CARD_AUTHOR", "Lukas Lepez — équipe InduSense")
    contact = os.getenv("INDUSENSE_MODEL_CARD_CONTACT", "lukas.lepez@alliance4u.fr")
    model_path = (report.get("artifacts") or {}).get("model", "modèle non renseigné")

    card = f"""---
language:
- fr
license: other
library_name: {"pytorch" if is_patchcore else "keras"}
pipeline_tag: image-classification
tags:
- anomaly-detection
- industrial-vision
- mvtec-ad
- bottle
datasets:
- mvtec-ad/bottle
metrics:
- roc_auc
- average_precision
- precision
- recall
- f1
---

# InduSense Vision — {model_label}

## Model Details

### Model Description

Ce modèle détecte les anomalies visuelles sur des images de bouteilles MVTec AD.
{"PatchCore compare les caractéristiques de chaque image à une banque de patchs sains." if is_patchcore else "L'auto-encodeur apprend à reconstruire uniquement des bouteilles saines ; une erreur de reconstruction élevée signale une anomalie."}
Le résultat est une aide au contrôle qualité et doit être validé par un opérateur.

- **Version :** {MODEL_CARD_VERSION}
- **Run :** `{report.get("run_id", run_dir.name)}`
- **Modèle :** {model_label}
- **Dataset :** `{report.get("dataset_version", "—")}`
- **Développé par :** {author}
- **Contact :** [{contact}](mailto:{contact})
- **Licence :** usage interne InduSense

## Uses

### Direct Use

Produire un score d'anomalie et une carte thermique pour une image de bouteille
prétraitée comme le dataset du run. Une alerte est déclenchée au-dessus du seuil
`{_number(threshold.get("value"), 8)}`, calibré au centile
{_number(threshold.get("percentile"), 1)} sur des images saines de validation.

### Downstream Use

Le score, la décision et la heatmap peuvent alimenter un tableau de contrôle qualité
ou prioriser une inspection humaine. Le système aval doit conserver la version du
modèle, le seuil et l'image source.

### Out-of-Scope Use

- Aucune décision automatique de rejet, d'arrêt de ligne ou de sécurité.
- Aucun usage sur une autre catégorie d'objet, caméra, éclairage ou résolution sans validation.
- Aucun diagnostic de la cause du défaut à partir du seul score ou de la heatmap.
- Aucun réglage du seuil sur le jeu de test ni apprentissage sur des images défectueuses.

## Bias, Risks, and Limitations

- Le modèle est validé uniquement sur MVTec AD `bottle`, dans des conditions visuelles contrôlées.
- Un changement d'éclairage, de cadrage, de fond ou de caméra peut créer de fausses alertes.
- Le test contient **{confusion.get("fn", "—")} défauts manqués** et
  **{confusion.get("fp", "—")} fausses alertes** au seuil retenu.
- Une heatmap peut réagir au contour ou au fond sans localiser correctement le défaut.
- L'absence d'alerte ne prouve pas qu'une pièce est conforme.
- Les performances peuvent varier selon le type, la taille et la position du défaut.

## Recommendations

- Faire valider chaque alerte et chaque rejet par un opérateur.
- Surveiller rappel, faux négatifs, fausses alertes, AUROC/AP image et métriques pixel.
- Recalibrer le seuil sur des images saines représentatives après tout changement de caméra.
- Constituer un jeu de validation industriel avant tout déploiement réel.
- Comparer l'auto-encodeur à une méthode fondée sur des caractéristiques préentraînées.

## How to Get Started with the Model

L'artefact du modèle est `{model_path}`. Les images doivent suivre le prétraitement
documenté dans le manifeste `{report.get("dataset_version", "—")}`. La décision est :

```python
predicted_anomaly = anomaly_score > {_number(threshold.get("value"), 12)}
```

## Training Details

- **Images saines d'entraînement :** {training.get("train_normal_images", "—")}
- **Images saines de validation :** {training.get("validation_normal_images", "—")}
- **Images de test :** {training.get("test_images", "—")}
- **Époques réalisées :** {training.get("epochs_completed", "—")}
- **Meilleure époque :** {training.get("best_epoch", "—")}
- **Augmentation :** {training.get("augmentation", "—")}
- **Entrée :** {_shape(architecture.get("input_shape"))}
- **Espace latent / carte de patchs :** {_shape(architecture.get("latent_shape"))}
- **Fonction de perte :** `{config.get("loss_name", "non applicable")}`
- **Graine aléatoire :** `{config.get("random_seed", reproducibility.get("random_seed", 42))}`

Les images saines de validation servent uniquement au calibrage du seuil. Le test,
qui contient des images saines et défectueuses, est réservé à l'évaluation finale.

## Evaluation

L'évaluation finale utilise exclusivement le split `test`. Le seuil ci-dessous est
calibré sur les images saines de validation, jamais sur les images de test.

| Mesure | Valeur |
|---|---:|
| Seuil retenu | {_number(threshold.get("value"), 8)} |
| Centile de calibration | {_number(threshold.get("percentile"), 1)} |
| AUROC image | {_number(image.get("auroc"))} |
| Average Precision image | {_number(image.get("average_precision"))} |
| Précision au seuil | {_number(image.get("precision"))} |
| Rappel au seuil | {_number(image.get("recall"))} |
| F1 au seuil | {_number(image.get("f1"))} |
| AUROC pixel | {_number(pixel.get("auroc"))} |
| Average Precision pixel | {_number(pixel.get("average_precision"))} |

Matrice de confusion : TN={confusion.get("tn", "—")}, FP={confusion.get("fp", "—")},
FN={confusion.get("fn", "—")}, TP={confusion.get("tp", "—")}.

**Analyse critique :** {report.get("critical_analysis", "Non renseignée.")}

## Environmental Impact

{_carbon_section(carbon)}

## Technical Specifications

- **Python :** `{reproducibility.get("python", "—")}`
- **Backend :** `{reproducibility.get("backend", reproducibility.get("device", "—"))}`
- **Keras :** `{reproducibility.get("keras", "non applicable")}`
- **Hash dataset :** `{report.get("dataset_hash", "—")}`
- **MLflow run :** `{report.get("mlflow_run_id") or "indisponible"}`
- **Date du run :** {report.get("created_at", "—")}

## Model Card Authors and Contact

Model card version {MODEL_CARD_VERSION}, rédigée par {author}. Questions, incidents
ou dérives : [{contact}](mailto:{contact}).
"""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def _carbon_section(carbon: dict[str, Any]) -> str:
    if carbon.get("available"):
        return (
            "- **Outil :** CodeCarbon\n"
            f"- **Pays :** `{carbon.get('country_iso_code', 'FRA')}`\n"
            f"- **Durée :** {_number(carbon.get('duration_seconds'), 2)} s\n"
            f"- **Énergie :** {_number(carbon.get('energy_kwh'), 6)} kWh\n"
            f"- **Émissions :** {_number(carbon.get('emissions_gco2eq'), 4)} gCO₂eq"
        )
    return (
        "- **Outil prévu :** CodeCarbon\n"
        "- **Mesure :** indisponible\n"
        f"- **Motif :** {carbon.get('reason', 'mesure non enregistrée pour ce run')}"
    )


def _number(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _shape(value: Any) -> str:
    return " × ".join(str(item) for item in value) if isinstance(value, list) else "—"
