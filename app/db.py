"""Persistencia: SQLAlchemy 2.0.

- Local: si no existe DATABASE_URL usa SQLite (fleet.db).
- Render + Neon: define DATABASE_URL como variable de entorno.

"""
from __future__ import annotations


import os
from collections.abc import Iterator

from sqlalchemy import Float, Index, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _normalize_url(url: str) -> str:
    """Neon entrega postgresql://...; con psycopg 3 hay que usar postgresql+psycopg://."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


DATABASE_URL = _normalize_url(os.environ.get("DATABASE_URL", "sqlite:///./fleet.db"))

if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs = {"connect_args": {"check_same_thread": False}}
else:
    _engine_kwargs = {
        "connect_args": {"connect_timeout": 15},  # no colgarse si la base tarda en despertar
        "pool_size": 5,
        "max_overflow": 5,
        "pool_recycle": 240,  # recicla conexiones antes de que Neon las suspenda (5 min)
    }

# pool_pre_ping=True: Neon suspende la base tras 5 min sin uso; así reconecta solo.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class TelemetryRow(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    robot_id: Mapped[str] = mapped_column(String(20), index=True)
    timestamp: Mapped[float] = mapped_column(Float)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    speed: Mapped[float] = mapped_column(Float)
    battery: Mapped[float] = mapped_column(Float)
    motor_temp: Mapped[float] = mapped_column(Float)
    vibration: Mapped[float] = mapped_column(Float)
    motor_current: Mapped[float] = mapped_column(Float)
    obstacle_distance: Mapped[float] = mapped_column(Float)
    # Se llenan en la etapa de /predict (nullable para no migrar la tabla después).
    predicted_status: Mapped[str | None] = mapped_column(String(10), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_telemetry_robot_ts", "robot_id", "timestamp"),)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "battery": self.battery,
            "motor_temp": self.motor_temp,
            "vibration": self.vibration,
            "motor_current": self.motor_current,
            "obstacle_distance": self.obstacle_distance,
            "predicted_status": self.predicted_status,
            "confidence": self.confidence,
        }


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
