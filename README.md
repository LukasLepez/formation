# Formation InduSense 4.0

Le projet est maintenant séparé en deux applications :

- `backend/` : API FastAPI, pipeline Python, Alembic, Docker Compose, données sources et artefacts.
- `frontend/` : interface Vite + React + MUI pour lancer les pipelines, suivre les logs live, générer les graphes et explorer Bronze/Silver/Gold.

## Lancer le backend

```powershell
cd backend
uv sync
uv run indusense-api
```

L'API écoute sur `http://127.0.0.1:8000`. Au démarrage, elle lance Docker Compose pour PostgreSQL et pgAdmin, sauf si `INDUSENSE_API_START_DOCKER=0`.

Si tu as déjà synchronisé `backend/.venv`, tu peux aussi lancer directement :

```powershell
cd backend
.\run-api.ps1
```

## Lancer le frontend

```powershell
cd frontend
npm run dev
```

L'interface écoute sur `http://127.0.0.1:5173` et proxyfie `/api` vers FastAPI.

## Fonctionnalités

- Lancer `bronze`, `silver`, `gold` ou toute la pipeline depuis React.
- Voir les logs live complets enregistrés dans `backend/artifacts/pipeline-runs/`.
- Naviguer dans les tables PostgreSQL Bronze, Silver et Gold.
- Générer les graphes Bronze/Silver depuis le frontend via FastAPI.
- Consulter l'historique des artefacts dans `backend/artifacts/ingestions/incidents/`.
- Stocker les Gold Datasets dans `backend/artifacts/gold-datasets/`.

## Commandes utiles

```powershell
cd backend
uv run build-gold-dataset --layer all
uv run build-gold-dataset --layer bronze
uv run build-gold-dataset --layer silver
uv run build-gold-dataset --layer gold
uv run build-gold-dataset --layer gold --no-db
```

```powershell
cd frontend
npm run build
```

## Base locale

PostgreSQL est exposé sur :

```text
postgresql+psycopg://postgres:postgres@localhost:5432/formation_indusense
```

pgAdmin est disponible sur `http://localhost:5050` avec `admin@example.com / admin`.

## Modèle de données

Le modèle SQLAlchemy est défini dans `backend/src/indusense/db/models.py`.
Les migrations Alembic sont dans `backend/alembic/versions/`.

La table `gold.gold_dataset` est écrite au format large : elle contient toutes
les colonnes du DataFrame Gold final, comme le CSV généré dans
`backend/artifacts/gold-datasets/`.
