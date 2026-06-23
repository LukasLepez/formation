"""Structure initiale Bronze, Silver et Gold.

Revision ID: 0001_structure_bronze_silver_gold
Revises:
Create Date: 2026-06-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_structure_bronze_silver_gold"
down_revision = None
branch_labels = None
depends_on = None

TYPE_COLUMNS = [
    "type_surchauffe",
    "type_baisse_pression",
    "type_vibration",
    "type_bruit_mecanique",
    "type_surconsommation",
    "type_blocage_mecanique",
    "type_alarme_capteur",
    "type_arret_urgence",
    "type_defaut_qualite",
]


def upgrade() -> None:
    """Crée les schémas et tables de base des trois couches."""

    op.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    op.execute("CREATE SCHEMA IF NOT EXISTS silver")
    op.execute("CREATE SCHEMA IF NOT EXISTS gold")

    op.create_table(
        "telemetry_raw",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("machine_id", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("pressure_bar", sa.Float(), nullable=True),
        sa.Column("voltage_mean_v", sa.Float(), nullable=True),
        sa.Column("rotation_mean_rpm", sa.Float(), nullable=True),
        sa.Column("pieces_produced", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        schema="bronze",
    )
    op.create_index("ix_bronze_telemetry_machine_time", "telemetry_raw", ["machine_id", "timestamp"], schema="bronze")

    incident_columns = [
        sa.Column("incident_id", sa.String(length=16), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("time", sa.Time(), nullable=False),
        sa.Column("operator_key", sa.String(length=64), nullable=False),
        sa.Column("machine_id", sa.String(length=16), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("shift", sa.String(length=32), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    ]
    incident_columns.extend(sa.Column(column, sa.Integer(), nullable=False, server_default="0") for column in TYPE_COLUMNS)
    op.create_table("incidents_raw", *incident_columns, schema="bronze")
    op.create_index("ix_bronze_incidents_machine", "incidents_raw", ["machine_id"], schema="bronze")

    op.create_table(
        "machine",
        sa.Column("machine_code", sa.String(length=16), primary_key=True),
        sa.Column("commissioning_date", sa.Date(), nullable=False),
        sa.Column("max_daily_capacity", sa.Integer(), nullable=False),
        sa.Column("max_hourly_capacity_pieces", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=32), nullable=False),
        sa.Column("production_line", sa.String(length=16), nullable=False),
        sa.Column("location", sa.String(length=16), nullable=False),
        sa.Column("criticality", sa.String(length=8), nullable=False),
        schema="bronze",
    )
    op.create_table(
        "maintenance",
        sa.Column("maintenance_id", sa.Integer(), primary_key=True),
        sa.Column("machine_code", sa.String(length=16), nullable=False),
        sa.Column("maintenance_at", sa.DateTime(), nullable=False),
        sa.Column("maintenance_type", sa.String(length=16), nullable=False),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column("component", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("related_incident_id", sa.String(length=16), nullable=True),
        sa.Column("duration_hours", sa.Numeric(precision=6, scale=2), nullable=False),
        schema="bronze",
    )
    op.create_index("ix_bronze_maintenance_machine_time", "maintenance", ["machine_code", "maintenance_at"], schema="bronze")

    op.create_table(
        "telemetry",
        sa.Column("machine_id", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("pressure_bar", sa.Float(), nullable=True),
        sa.Column("voltage_mean_v", sa.Float(), nullable=True),
        sa.Column("rotation_mean_rpm", sa.Float(), nullable=True),
        sa.Column("pieces_produced", sa.Integer(), nullable=True),
        sa.Column("machine_id_std", sa.String(length=16), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("machine_id_std", "timestamp"),
        schema="silver",
    )
    op.create_index("ix_silver_telemetry_machine_window", "telemetry", ["machine_id_std", "window_start"], schema="silver")

    silver_incident_columns = [
        sa.Column("incident_id", sa.String(length=16), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("time", sa.Time(), nullable=False),
        sa.Column("operator_key", sa.String(length=64), nullable=False),
        sa.Column("machine_id", sa.String(length=16), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("shift", sa.String(length=32), nullable=True),
        sa.Column("incident_at", sa.DateTime(), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.Column("machine_id_std", sa.String(length=16), nullable=False),
    ]
    silver_incident_columns.extend(sa.Column(column, sa.Integer(), nullable=False, server_default="0") for column in TYPE_COLUMNS)
    op.create_table("incidents", *silver_incident_columns, schema="silver")
    op.create_index("ix_silver_incidents_machine_window", "incidents", ["machine_id_std", "window_start"], schema="silver")

    op.create_table(
        "machine",
        sa.Column("machine_code", sa.String(length=16), primary_key=True),
        sa.Column("commissioning_date", sa.Date(), nullable=False),
        sa.Column("max_daily_capacity", sa.Integer(), nullable=False),
        sa.Column("max_hourly_capacity_pieces", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=32), nullable=False),
        sa.Column("production_line", sa.String(length=16), nullable=False),
        sa.Column("location", sa.String(length=16), nullable=False),
        sa.Column("criticality", sa.String(length=8), nullable=False),
        schema="silver",
    )
    op.create_table(
        "maintenance",
        sa.Column("maintenance_id", sa.Integer(), primary_key=True),
        sa.Column("machine_code", sa.String(length=16), nullable=False),
        sa.Column("maintenance_at", sa.DateTime(), nullable=False),
        sa.Column("maintenance_type", sa.String(length=16), nullable=False),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column("component", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("related_incident_id", sa.String(length=16), nullable=True),
        sa.Column("duration_hours", sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column("maintenance_hour", sa.DateTime(), nullable=True),
        schema="silver",
    )

    op.create_table(
        "gold_dataset",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("machine_id_std", sa.String(length=16), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.Column("window_end", sa.DateTime(), nullable=False),
        sa.Column("split_set", sa.String(length=16), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=True),
        sa.Column("labels", postgresql.JSONB(), nullable=True),
        schema="gold",
    )


def downgrade() -> None:
    """Supprime les tables et schémas créés par la migration."""

    op.drop_table("gold_dataset", schema="gold")
    op.drop_table("maintenance", schema="silver")
    op.drop_table("machine", schema="silver")
    op.drop_index("ix_silver_incidents_machine_window", table_name="incidents", schema="silver")
    op.drop_table("incidents", schema="silver")
    op.drop_index("ix_silver_telemetry_machine_window", table_name="telemetry", schema="silver")
    op.drop_table("telemetry", schema="silver")
    op.drop_index("ix_bronze_maintenance_machine_time", table_name="maintenance", schema="bronze")
    op.drop_table("maintenance", schema="bronze")
    op.drop_table("machine", schema="bronze")
    op.drop_index("ix_bronze_incidents_machine", table_name="incidents_raw", schema="bronze")
    op.drop_table("incidents_raw", schema="bronze")
    op.drop_index("ix_bronze_telemetry_machine_time", table_name="telemetry_raw", schema="bronze")
    op.drop_table("telemetry_raw", schema="bronze")
    op.execute("DROP SCHEMA IF EXISTS gold")
    op.execute("DROP SCHEMA IF EXISTS silver")
    op.execute("DROP SCHEMA IF EXISTS bronze")
