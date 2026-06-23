# Formation InduSense 4.0

## Construire les couches Bronze, Silver et Gold

Le pipeline crée maintenant la base analytique complète :

- `bronze` : données sources typées mais non nettoyées (`telemetry_raw`, `incidents_raw`, `machine`, `maintenance`).
- `silver` : données nettoyées et normalisées, lues depuis `bronze`.
- `gold` : dataset machine-heure final, lu depuis `silver`, avec features et labels.

Les couches `bronze` et `silver` ne persistent pas les identifiants directs des
opérateurs : `operator_name` et `operator_badge` ne sont pas chargés en base.
Une clé `operator_key` aléatoire et non réversible est générée en mémoire pour
le run. Le champ métier `comment` est conservé en Bronze/Silver, mais il n'est
pas utilisé dans les features Gold actuelles.

Par défaut, la commande exécute les trois couches dans l'ordre :

```powershell
uv run build-gold-dataset
```

Équivalent explicite :

```powershell
uv run build-gold-dataset --layer all
```

On peut aussi relancer une seule couche si ses dépendances existent déjà en base :

```powershell
uv run build-gold-dataset --layer bronze
uv run build-gold-dataset --layer silver
uv run build-gold-dataset --layer gold
```

`--stage` est accepté comme alias de `--layer`.

La couche Gold respecte le contrat de `docs/gold_dataset.md` : une ligne par
machine et par heure, fenêtres glissantes intra-machine, labels par lookahead,
split temporel 70/15/15, et statistiques de normalisation fittées uniquement
sur le train set. Elle produit un CSV de traçabilité dans `gold-dataset/` :

```text
gold-dataset/gold_dataset_YYYYMMDDHHMMSS.csv
```

La commande démarre aussi le stack Docker Compose du projet si nécessaire, puis
applique les migrations Alembic et remplit les schémas PostgreSQL :

```powershell
postgresql+psycopg://postgres:postgres@localhost:5432/formation_indusense
```

pgAdmin reste disponible sur `http://localhost:5050` avec
`admin@example.com / admin`.

Pour generer uniquement le CSV sans PostgreSQL :

```powershell
uv run build-gold-dataset --layer gold --no-db
```

Dans Python, le dataset Gold persisté peut être rechargé sous forme de
DataFrame large avec :

```python
from indusense.db.gold_loader import load_gold_from_db

gold = load_gold_from_db()
```

## Modèle de données

Le modèle SQLAlchemy est défini dans `src/indusense/db/models.py`.
Les migrations Alembic sont dans `alembic/versions/`.

La table `gold.gold_dataset` garde une structure stable :

- colonnes d'identité : `machine_id_std`, `window_start`, `window_end`, `split_set`;
- `features` en JSONB pour les variables explicatives;
- `labels` en JSONB pour les cibles `label_failure_next_*` et compteurs futurs.

Le loader `load_gold_from_db()` reconstruit automatiquement le DataFrame large
attendu par les notebooks ou l'entraînement.
