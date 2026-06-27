"""Chargeur pratique pour la table canonique du Gold Dataset."""

from __future__ import annotations

import pandas as pd

from indusense.processing.ingestion import load_gold_from_db as _load_gold_from_db


def load_gold_from_db(table_name: str = "gold_dataset", database_url: str | None = None) -> pd.DataFrame:
    """Charge le Gold Dataset persisté dans PostgreSQL.

    C'est le nom de fonction public utilisé dans ``docs/gold_dataset.md``.
    """

    return _load_gold_from_db(database_url=database_url, table_name=table_name)


def load_gold(table_name: str = "gold_dataset", database_url: str | None = None) -> pd.DataFrame:
    """Charge le Gold Dataset persisté dans PostgreSQL."""

    return load_gold_from_db(database_url=database_url, table_name=table_name)
