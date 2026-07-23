# InduSense Backend

Backend FastAPI et pipeline Python du projet InduSense.

```powershell
uv sync
uv run indusense-api
```

## TP B6 — auto-encodeur

1. Ouvrir **Données images** et préparer le jeu MVTec AD bottle.
2. Ouvrir **Auto-encodeur**, choisir MSE ou SSIM, puis lancer une exécution.
3. Consulter la courbe d'apprentissage, le seuil calibré sur les seules images
   saines de validation, les AUROC image/pixel, la matrice de confusion et les
   heatmaps comparées aux masques.

Le notebook équivalent se trouve dans `notebooks/tp_deep_learning.ipynb`. Les
exécutions, modèles, figures, rapports et données MLflow sont écrits dans
`artifacts/vision-model-runs/`.

Le projet utilise l'API Keras 3 avec le backend PyTorch : TensorFlow 2.21 ne
fournit pas de roue Python 3.14, qui est la version Python de ce backend.

L'API écoute sur `http://127.0.0.1:8000`.

## TP B7 — pipelines rejouables

Les résultats sont produits dans `artifacts/` (ignoré par Git). Après `uv sync` :

```powershell
uv run python scripts/run_maintenance_pipeline.py --horizon 24 --tune --n-trials 15 --study-mode frugal
uv run python scripts/run_vision_pipeline.py --epochs 30 --img-size 128 --batch-size 32
```

La page Maintenance ML permet de choisir une étude Optuna frugale (pruning) ou
lourde, puis de consulter PR-AUC, SHAP, MLflow et CodeCarbon.
