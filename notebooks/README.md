# Notebooks incidents

Ce dossier contient les notebooks a executer pour creer les bases et generer les graphes.

## Ordre conseille

1. Executer `creation_bronze.ipynb`

   Ce notebook demarre PostgreSQL avec Docker si besoin, lit les fichiers de `data/`, anonymise les operateurs et cree la base `bronze`.

2. Choisir la suite selon le besoin :

   - Pour generer les graphes directement sur les donnees bronze, executer `graphes_rapports.ipynb` avec `SOURCE_LAYER = "bronze"`.
   - Pour preparer les donnees nettoyees, executer `creation_silver.ipynb`. Il lit la base `bronze`, applique le dedoublonnage et le traitement des valeurs manquantes, puis cree la base `silver`.

3. Pour generer les graphes sur les donnees silver, executer `graphes_rapports.ipynb` avec `SOURCE_LAYER = "silver"`.

## Variable a changer pour les graphes

Dans `graphes_rapports.ipynb`, la variable se trouve dans la premiere cellule de code :

```python
SOURCE_LAYER = "bronze"
```

Valeurs possibles :

- `"bronze"` : graphes sur la base bronze.
- `"silver"` : graphes sur la base silver, apres execution de `creation_silver.ipynb`.

## Sorties generees

- Les runs bronze sont ecrits dans `artifacts/ingestions/incidents/<timestamp>_bronze/`.
- Les runs silver sont ecrits dans `artifacts/ingestions/incidents/<timestamp>_silver/`.
- Les rapports de graphes sont ecrits dans `artifacts/ingestions/incidents/<timestamp>_<bronze|silver>_report/`.
