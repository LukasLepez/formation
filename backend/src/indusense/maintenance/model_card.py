"""Génération de la model card Hugging Face du modèle de maintenance."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


MODEL_CARD_VERSION = "1.0.0"
DEFAULT_AUTHOR = "Lukas Lepez — équipe InduSense"
DEFAULT_CONTACT = "lukas.lepez@alliance4u.fr"

MODEL_LABELS = {
    "logistic_regression": "régression logistique",
    "decision_tree": "arbre de décision",
    "random_forest": "Random Forest",
    "random_forest_balanced": "Random Forest rééquilibrée",
    "xgboost": "XGBoost",
}


def generate_maintenance_model_card(
    report: dict[str, Any],
    run_dir: Path,
    *,
    model_version: str = MODEL_CARD_VERSION,
    registry_name: str | None = None,
    registry_stage: str | None = None,
) -> Path:
    """Crée un ``README.md`` conforme aux sections standard Hugging Face."""

    results = report.get("results") or []
    if not results:
        raise ValueError("La model card requiert au moins un résultat d'évaluation.")
    best_model = report.get("best_model")
    best = next((item for item in results if item.get("model") == best_model), results[0])
    model_name = str(best.get("model") or best_model or "modèle inconnu")
    model_path = _posix_path(best.get("model_path") or f"models/{model_name}.pkl")
    label_column = str(report.get("label_column") or "cible non renseignée")
    horizon = _prediction_horizon(label_column)
    horizon_text = f"les **{horizon} h** à venir" if horizon else "la fenêtre prédictive configurée"
    author = os.getenv("INDUSENSE_MODEL_CARD_AUTHOR", DEFAULT_AUTHOR)
    contact = os.getenv("INDUSENSE_MODEL_CARD_CONTACT", DEFAULT_CONTACT)
    threshold = _number(best.get("threshold"), 6)
    class_balance = report.get("class_balance") or {}
    confusion = best.get("test_confusion_matrix") or {}
    carbon = report.get("carbon") or {}
    reproducibility = report.get("reproducibility") or {}
    selected_models = report.get("selected_models") or [item.get("model") for item in results]
    model_list = ", ".join(MODEL_LABELS.get(str(item), str(item)) for item in selected_models if item)
    registry_lines = ""
    if registry_name:
        registry_lines += f"- **Registry MLflow :** `{registry_name}`\n"
    if registry_stage:
        registry_lines += f"- **Stage MLflow :** `{registry_stage}`\n"

    front_matter = _front_matter(report, best)
    environmental_impact = _environmental_impact(carbon)
    card = f"""{front_matter}

# InduSense Maintenance Predictive

## Model Details

### Model Description

Ce modèle de classification binaire estime, à partir des relevés industriels et de
l'historique récent, si une machine risque de tomber en panne dans {horizon_text}.
Il fournit une aide à la priorisation des contrôles de maintenance ; il ne remplace
pas le diagnostic ni la décision d'un technicien.

- **Version de la model card :** {MODEL_CARD_VERSION}
- **Version du modèle :** {model_version}
- **Run applicatif :** `{report.get("run_id", run_dir.name)}`
- **Modèle retenu :** `{model_name}`
{registry_lines}- **Développé par :** {author}
- **Contact :** [{contact}](mailto:{contact})
- **Type :** classification tabulaire supervisée
- **Langue de la documentation :** français
- **Licence :** usage interne InduSense ; aucune licence de redistribution accordée

## Uses

### Direct Use

Le modèle peut produire un score de risque pour chaque observation conforme au schéma
du Gold Dataset InduSense. Au seuil **{threshold}**, ce score peut alimenter une liste
d'alertes à examiner par un technicien de maintenance.

### Downstream Use

Le score peut être intégré dans un tableau de bord, un système de priorisation des
inspections ou un outil d'aide à la planification. Le système aval doit conserver le
score, le seuil, la version du modèle et prévoir une validation humaine.

### Out-of-Scope Use

- Aucune décision automatique d'arrêt, de réparation ou de remplacement de machine.
- Aucun usage sur un autre parc, d'autres capteurs ou une autre définition de panne
  sans nouvelle validation.
- Aucun usage pour la sécurité des personnes ou une fonction critique en temps réel.
- Aucune interprétation du score comme une certitude de panne ou de non-panne.
- Aucun réentraînement sur les jeux validation/test ni découpage aléatoire des données.

## Bias, Risks, and Limitations

- Les pannes sont minoritaires dans le train ({_percent(class_balance.get("train_positive_rate"))}) :
  le signal est déséquilibré et l'accuracy seule serait trompeuse.
- Les relations apprises reflètent uniquement les machines, périodes, capteurs et
  pratiques de maintenance contenus dans `{report.get("gold_run_name", "dataset inconnu")}`.
- Une dérive des capteurs, du parc ou des pratiques opérationnelles peut dégrader les
  performances après déploiement.
- Les valeurs manquantes sont imputées ; un volume inhabituel de données manquantes
  peut masquer une anomalie réelle.
- Au seuil retenu, le test contient **{confusion.get("fn", "—")} faux négatifs**
  (pannes manquées) et **{confusion.get("fp", "—")} faux positifs** (alertes inutiles).
- La PR-AUC test ne garantit pas la même performance sur une période ou un site futur.

## Recommendations

- Faire examiner chaque alerte par un technicien et ne jamais automatiser une décision
  de maintenance à partir du seul score.
- Surveiller la PR-AUC, le rappel, les faux négatifs, les faux positifs, la dérive des
  variables et le taux de valeurs manquantes.
- Revalider le seuil selon le coût métier comparé d'une panne manquée et d'une fausse
  alerte ; la stratégie actuelle est `{report.get("threshold_strategy", "non renseignée")}`.
- Réévaluer le modèle sur une fenêtre temporelle récente avant tout changement de site,
  de machine, de capteur ou de définition de la cible.
- Conserver une solution de repli et permettre aux opérateurs de contester une alerte.

## How to Get Started with the Model

Le fichier sérialisé `{model_path}` contient le pipeline complet (prétraitement puis
classifieur). L'entrée doit fournir les mêmes **{report.get("features", "—")} variables**
que le Gold Dataset du run.

```python
import pickle
import pandas as pd

with open("{model_path}", "rb") as model_file:
    model = pickle.load(model_file)

scores = model.predict_proba(pd.DataFrame(observations))[:, 1]
alerts = scores >= {best.get("threshold", 0.5)}
```

Ne chargez un fichier pickle que depuis une source de confiance.

## Training Details

### Training Data

- **Source :** Gold Dataset InduSense `{report.get("gold_run_name", "—")}`
- **Volume total :** {_integer(report.get("rows"))} observations
- **Variables utilisées :** {_integer(report.get("features"))}
- **Cible :** `{label_column}`
- **Découpage :** `split_set` temporel train / validation / test
- **Hash du dataset :** `{reproducibility.get("dataset_hash", "—")}`

Les identifiants, dates, autres labels et variables décrivant le futur sont exclus afin
de limiter les fuites de données.

### Training Procedure

Les modèles comparés sont : {model_list or "non renseignés"}. L'imputation numérique
utilise la médiane, les catégories sont imputées par la valeur la plus fréquente puis
encodées en one-hot, et les variables numériques sont standardisées pour la régression
logistique. Le déséquilibre est compensé selon le modèle par des poids de classe ou
`scale_pos_weight`. Une validation croisée `TimeSeriesSplit` est réalisée sur le train.
Le meilleur candidat est sélectionné par PR-AUC validation et le seuil est choisi sur
la validation, jamais sur le test.

## Evaluation

### Testing Data, Factors & Metrics

L'évaluation finale utilise exclusivement le split temporel `test`. La métrique
principale est la PR-AUC, adaptée aux pannes rares. ROC-AUC mesure la séparation
globale ; précision, rappel et F1 décrivent le comportement au seuil retenu.

| Mesure | Validation | Test |
|---|---:|---:|
| PR-AUC | {_number(best.get("pr_auc_validation"))} | {_number(best.get("pr_auc_test"))} |
| ROC-AUC | {_number(best.get("roc_auc_validation"))} | {_number(best.get("roc_auc_test"))} |
| Précision | {_number(best.get("precision_validation"))} | {_number(best.get("precision_test"))} |
| Rappel | {_number(best.get("recall_validation"))} | {_number(best.get("recall_test"))} |
| F1 | {_number(best.get("f1_validation"))} | {_number(best.get("f1_test"))} |
| Seuil retenu | {threshold} | {threshold} |

### Results

Matrice de confusion sur le test au seuil retenu :

| Vrais négatifs | Faux positifs | Faux négatifs | Vrais positifs |
|---:|---:|---:|---:|
| {confusion.get("tn", "—")} | {confusion.get("fp", "—")} | {confusion.get("fn", "—")} | {confusion.get("tp", "—")} |

La performance doit être interprétée avec les coûts métier : les faux négatifs
correspondent à des pannes manquées, les faux positifs à des inspections inutiles.
Le coût configuré est FN × {report.get("false_negative_cost", 20)} et
FP × {report.get("false_positive_cost", 1)}.

## Environmental Impact

{environmental_impact}

## Technical Specifications

- **Framework :** scikit-learn / XGBoost selon le candidat retenu
- **Suivi d'expériences :** MLflow, stockage SQLite local
- **Format du modèle :** pickle Python
- **Reproductibilité :** graine pseudo-aléatoire `{report.get("random_state", 42)}`
- **Python :** `{reproducibility.get("python_version", "—")}`
- **scikit-learn :** `{reproducibility.get("sklearn_version", "—")}`
- **Date du run :** {report.get("created_at", "—")}

## Model Card Authors and Contact

Model card version {MODEL_CARD_VERSION}, rédigée par {author}. Pour les questions,
incidents, demandes d'audit ou signalements de dérive :
[{contact}](mailto:{contact}).
"""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def _front_matter(report: dict[str, Any], best: dict[str, Any]) -> str:
    dataset = report.get("gold_run_name", "private-indusense-gold-dataset")
    return f"""---
language:
- fr
license: other
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
- predictive-maintenance
- imbalanced-classification
- time-series-split
- indusense
datasets:
- {dataset}
metrics:
- pr_auc
- roc_auc
- precision
- recall
- f1
model-index:
- name: InduSense Maintenance Predictive
  results:
  - task:
      type: tabular-classification
      name: Prédiction de panne
    dataset:
      name: {dataset}
      type: private
      split: test
    metrics:
    - type: pr_auc
      value: {_number(best.get("pr_auc_test"))}
      name: PR-AUC test
    - type: roc_auc
      value: {_number(best.get("roc_auc_test"))}
      name: ROC-AUC test
    - type: f1
      value: {_number(best.get("f1_test"))}
      name: F1 test
---"""


def _environmental_impact(carbon: dict[str, Any]) -> str:
    if carbon.get("available"):
        return (
            "- **Outil :** CodeCarbon\n"
            f"- **Durée mesurée :** {_number(carbon.get('duration_seconds'), 2)} s\n"
            f"- **Énergie consommée :** {_number(carbon.get('energy_kwh'), 6)} kWh\n"
            f"- **Émissions estimées :** {_number(carbon.get('emissions_gco2eq'), 4)} gCO₂eq\n\n"
            "La mesure couvre l'entraînement et l'évaluation de ce run dans "
            "l'environnement d'exécution indiqué."
        )
    reason = carbon.get("reason") or "CodeCarbon n'a pas fourni de mesure exploitable."
    return (
        "- **Outil prévu :** CodeCarbon\n"
        "- **Mesure :** indisponible pour ce run\n"
        f"- **Motif :** {reason}\n\n"
        "Ce run ne doit pas être utilisé pour déclarer une empreinte carbone chiffrée. "
        "Relancer l'entraînement avec CodeCarbon actif pour obtenir une mesure traçable."
    )


def _prediction_horizon(label_column: str) -> str | None:
    match = re.search(r"next_(\d+)h", label_column)
    return match.group(1) if match else None


def _number(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f} %"
    except (TypeError, ValueError):
        return "non renseigné"


def _integer(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _posix_path(value: Any) -> str:
    return str(value).replace("\\", "/")
