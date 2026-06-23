# Exécuter le pipeline InduSense

Cette page explique comment lancer le script de génération des couches `bronze`, `silver` et `gold`.

## Prérequis

- Se placer à la racine du projet :

```powershell
cd D:\Code\formation
```

- Avoir `uv` installé.
- Avoir Docker Desktop disponible si le pipeline doit écrire dans PostgreSQL.

Par défaut, la base locale utilisée est :

```text
postgresql+psycopg://postgres:postgres@localhost:5432/formation_indusense
```

Le script démarre automatiquement PostgreSQL et pgAdmin via Docker Compose, sauf si l’option `--no-docker` est utilisée.

## Commande recommandée

Pour générer toute la chaîne de données, lancer :

```powershell
uv run build-gold-dataset
```

Cette commande est équivalente à :

```powershell
uv run build-gold-dataset --layer all
```

Elle exécute les couches dans cet ordre :

1. `bronze` : charge les fichiers sources dans PostgreSQL.
2. `silver` : lit `bronze`, nettoie les données et écrit `silver`.
3. `gold` : lit `silver`, construit les features/labels et écrit `gold`.

## Données personnelles et anonymisation

Le pipeline ne charge pas en base les données qui peuvent identifier directement une personne.

Dans les couches `bronze` et `silver` :

- `operator_name` n’est pas persisté;
- `operator_badge` n’est pas persisté;
- seule une clé `operator_key` aléatoire et non réversible est conservée pour distinguer les opérateurs au sein d’un run.

Le mapping entre badge source et `operator_key` n’est pas exporté. Il est généré en mémoire pendant l’exécution puis oublié.

Le champ métier `comment` est conservé en Bronze/Silver pour audit et analyse qualité. Il n’est pas utilisé par le Gold Dataset actuel, qui s’appuie sur les compteurs d’incidents, la sévérité et les colonnes `type_*`.

## Exécuter une seule couche

### Bronze uniquement

```powershell
uv run build-gold-dataset --layer bronze
```

Cette commande lit :

- `data/telemetry.csv`
- `data/releves_incidents.csv`
- `data/machine.sql`

Puis elle remplit les tables du schéma PostgreSQL `bronze`.

### Silver uniquement

```powershell
uv run build-gold-dataset --layer silver
```

Cette commande suppose que la couche `bronze` existe déjà en base. Elle lit `bronze`, applique le nettoyage, puis remplit le schéma `silver`.

### Gold uniquement

```powershell
uv run build-gold-dataset --layer gold
```

Cette commande suppose que la couche `silver` existe déjà en base. Elle construit le dataset final, écrit la table `gold.gold_dataset` et génère un CSV de traçabilité dans `gold-dataset/`.

## Alias `--stage`

`--stage` peut être utilisé à la place de `--layer` :

```powershell
uv run build-gold-dataset --stage all
uv run build-gold-dataset --stage bronze
uv run build-gold-dataset --stage silver
uv run build-gold-dataset --stage gold
```

## Générer uniquement le CSV Gold sans PostgreSQL

Pour construire le dataset Gold depuis les fichiers sources, sans écrire en base :

```powershell
uv run build-gold-dataset --layer gold --no-db
```

Cette commande génère uniquement un fichier CSV :

```text
gold-dataset/gold_dataset_YYYYMMDDHHMMSS.csv
```

Elle est pratique pour vérifier rapidement le pipeline sans Docker/PostgreSQL.

## Options utiles

### Ne pas démarrer Docker automatiquement

Si PostgreSQL est déjà lancé :

```powershell
uv run build-gold-dataset --layer all --no-docker
```

### Utiliser une autre base

```powershell
uv run build-gold-dataset --layer all --database-url "postgresql+psycopg://user:password@host:5432/database"
```

### Changer les chemins sources

```powershell
uv run build-gold-dataset `
  --telemetry data/telemetry.csv `
  --incidents data/releves_incidents.csv `
  --machine-sql data/machine.sql
```

### Changer le dossier de sortie CSV

```powershell
uv run build-gold-dataset --layer gold --output-dir gold-dataset
```

### Logs plus détaillés

```powershell
uv run build-gold-dataset --layer all --log-level DEBUG
```

## Résultat attendu

Après une exécution complète, PostgreSQL contient :

- `bronze.telemetry_raw`
- `bronze.incidents_raw`
- `bronze.machine`
- `bronze.maintenance`
- `silver.telemetry`
- `silver.incidents`
- `silver.machine`
- `silver.maintenance`
- `gold.gold_dataset`

La couche Gold produit aussi un CSV de traçabilité :

```text
gold-dataset/gold_dataset_YYYYMMDDHHMMSS.csv
```

Le dataset Gold peut être rechargé en Python avec :

```python
from indusense.db.gold_loader import load_gold_from_db

gold = load_gold_from_db()
print(gold.shape)
```

## Dépannage

### Docker ne démarre pas

Lancer Docker Desktop manuellement, puis relancer :

```powershell
uv run build-gold-dataset --layer all
```

### Silver échoue

Vérifier que `bronze` a déjà été généré :

```powershell
uv run build-gold-dataset --layer bronze
uv run build-gold-dataset --layer silver
```

### Gold échoue

Vérifier que `silver` existe déjà :

```powershell
uv run build-gold-dataset --layer silver
uv run build-gold-dataset --layer gold
```

### Tester sans base

Pour isoler les problèmes PostgreSQL/Docker :

```powershell
uv run build-gold-dataset --layer gold --no-db
```
