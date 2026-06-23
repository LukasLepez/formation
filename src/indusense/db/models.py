"""Modèle de données SQLAlchemy des couches Bronze, Silver et Gold."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, Date, DateTime, Float, Integer, MetaData, Numeric, String, Text, Time
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


convention_nommage = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base déclarative commune aux tables InduSense."""

    metadata = MetaData(naming_convention=convention_nommage)


class BronzeTelemetryRaw(Base):
    """Mesures de télémétrie brutes telles que lues depuis le CSV source."""

    __tablename__ = "telemetry_raw"
    __table_args__ = {"schema": "bronze"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float)
    pressure_bar: Mapped[Optional[float]] = mapped_column(Float)
    voltage_mean_v: Mapped[Optional[float]] = mapped_column(Float)
    rotation_mean_rpm: Mapped[Optional[float]] = mapped_column(Float)
    pieces_produced: Mapped[Optional[int]] = mapped_column(Integer)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BronzeIncidentRaw(Base):
    """Incident brut déclaré par un opérateur."""

    __tablename__ = "incidents_raw"
    __table_args__ = {"schema": "bronze"}

    incident_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[time] = mapped_column(Time, nullable=False)
    operator_key: Mapped[str] = mapped_column(String(64), nullable=False)
    machine_id: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    shift: Mapped[Optional[str]] = mapped_column(String(32))
    type_surchauffe: Mapped[int] = mapped_column(Integer, nullable=False)
    type_baisse_pression: Mapped[int] = mapped_column(Integer, nullable=False)
    type_vibration: Mapped[int] = mapped_column(Integer, nullable=False)
    type_bruit_mecanique: Mapped[int] = mapped_column(Integer, nullable=False)
    type_surconsommation: Mapped[int] = mapped_column(Integer, nullable=False)
    type_blocage_mecanique: Mapped[int] = mapped_column(Integer, nullable=False)
    type_alarme_capteur: Mapped[int] = mapped_column(Integer, nullable=False)
    type_arret_urgence: Mapped[int] = mapped_column(Integer, nullable=False)
    type_defaut_qualite: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BronzeMachine(Base):
    """Référentiel machine brut extrait du fichier SQL."""

    __tablename__ = "machine"
    __table_args__ = {"schema": "bronze"}

    machine_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    commissioning_date: Mapped[date] = mapped_column(Date, nullable=False)
    max_daily_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    max_hourly_capacity_pieces: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(32), nullable=False)
    production_line: Mapped[str] = mapped_column(String(16), nullable=False)
    location: Mapped[str] = mapped_column(String(16), nullable=False)
    criticality: Mapped[str] = mapped_column(String(8), nullable=False)


class BronzeMaintenance(Base):
    """Intervention de maintenance brute extraite du fichier SQL."""

    __tablename__ = "maintenance"
    __table_args__ = {"schema": "bronze"}

    maintenance_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    machine_code: Mapped[str] = mapped_column(String(16), nullable=False)
    maintenance_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    maintenance_type: Mapped[str] = mapped_column(String(16), nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    related_incident_id: Mapped[Optional[str]] = mapped_column(String(16))
    duration_hours: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)


class SilverTelemetry(Base):
    """Télémétrie nettoyée : dédoublonnée, typée et normalisée."""

    __tablename__ = "telemetry"
    __table_args__ = {"schema": "silver"}

    machine_id: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float)
    pressure_bar: Mapped[Optional[float]] = mapped_column(Float)
    voltage_mean_v: Mapped[Optional[float]] = mapped_column(Float)
    rotation_mean_rpm: Mapped[Optional[float]] = mapped_column(Float)
    pieces_produced: Mapped[Optional[int]] = mapped_column(Integer)
    machine_id_std: Mapped[str] = mapped_column(String(16), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SilverIncident(Base):
    """Incident enrichi avec l'horodatage et la fenêtre horaire de jointure."""

    __tablename__ = "incidents"
    __table_args__ = {"schema": "silver"}

    incident_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[time] = mapped_column(Time, nullable=False)
    operator_key: Mapped[str] = mapped_column(String(64), nullable=False)
    machine_id: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    shift: Mapped[Optional[str]] = mapped_column(String(32))
    incident_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    machine_id_std: Mapped[str] = mapped_column(String(16), nullable=False)
    type_surchauffe: Mapped[int] = mapped_column(Integer, nullable=False)
    type_baisse_pression: Mapped[int] = mapped_column(Integer, nullable=False)
    type_vibration: Mapped[int] = mapped_column(Integer, nullable=False)
    type_bruit_mecanique: Mapped[int] = mapped_column(Integer, nullable=False)
    type_surconsommation: Mapped[int] = mapped_column(Integer, nullable=False)
    type_blocage_mecanique: Mapped[int] = mapped_column(Integer, nullable=False)
    type_alarme_capteur: Mapped[int] = mapped_column(Integer, nullable=False)
    type_arret_urgence: Mapped[int] = mapped_column(Integer, nullable=False)
    type_defaut_qualite: Mapped[int] = mapped_column(Integer, nullable=False)


class SilverMachine(Base):
    """Référentiel machine repris dans la couche Silver."""

    __tablename__ = "machine"
    __table_args__ = {"schema": "silver"}

    machine_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    commissioning_date: Mapped[date] = mapped_column(Date, nullable=False)
    max_daily_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    max_hourly_capacity_pieces: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(32), nullable=False)
    production_line: Mapped[str] = mapped_column(String(16), nullable=False)
    location: Mapped[str] = mapped_column(String(16), nullable=False)
    criticality: Mapped[str] = mapped_column(String(8), nullable=False)


class SilverMaintenance(Base):
    """Maintenance Silver avec fenêtre horaire calculée."""

    __tablename__ = "maintenance"
    __table_args__ = {"schema": "silver"}

    maintenance_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    machine_code: Mapped[str] = mapped_column(String(16), nullable=False)
    maintenance_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    maintenance_type: Mapped[str] = mapped_column(String(16), nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    related_incident_id: Mapped[Optional[str]] = mapped_column(String(16))
    duration_hours: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    maintenance_hour: Mapped[Optional[datetime]] = mapped_column(DateTime)


class GoldDataset(Base):
    """Table contractuelle Gold, une ligne par machine et par heure.

    La persistance opérationnelle du Gold Dataset conserve toutes les colonnes
    du DataFrame final. Les champs JSON documentent le modèle cible Alembic
    minimal quand on veut suivre la structure logique avec SQLAlchemy.
    """

    __tablename__ = "gold_dataset"
    __table_args__ = {"schema": "gold"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    machine_id_std: Mapped[str] = mapped_column(String(16), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    split_set: Mapped[str] = mapped_column(String(16), nullable=False)
    features: Mapped[Optional[dict]] = mapped_column(JSONB)
    labels: Mapped[Optional[dict]] = mapped_column(JSONB)
