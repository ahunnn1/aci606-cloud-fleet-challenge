"""Métricas operacionales en memoria (seguras entre hilos).

Los contadores viven en el proceso: se reinician cuando el servicio se reinicia o
se "duerme" (Render free). Lo que debe sobrevivir (telemetría, predicciones) está en
la base de datos y se consulta aparte en /metrics.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """Percentil por rango más cercano (q entre 0 y 100)."""
    if not sorted_values:
        return None
    k = max(0, min(len(sorted_values) - 1, int(round(q / 100 * len(sorted_values) + 0.5)) - 1))
    return sorted_values[k]


class Metrics:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.monotonic()
        self.window = window
        self._reset()

    def _reset(self) -> None:
        self.total = 0
        self.by_route: Counter[str] = Counter()
        self.by_status: Counter[str] = Counter()
        self.errors_4xx = 0
        self.errors_5xx = 0
        self.telemetry_requests = 0
        self.telemetry_accepted = 0
        self.predict_requests = 0
        self.predict_errors = 0
        self.predict_latency_ms: deque[float] = deque(maxlen=self.window)
        self.inference_count = 0
        self.inference_sum_ms = 0.0
        self.inference_max_ms = 0.0

    # --- escritura -------------------------------------------------------------------
    def record_request(self, method: str, route: str, status: int, latency_ms: float) -> None:
        key = f"{method} {route}"
        with self._lock:
            self.total += 1
            self.by_route[key] += 1
            self.by_status[str(status)] += 1
            if 400 <= status < 500:
                self.errors_4xx += 1
            elif status >= 500:
                self.errors_5xx += 1
            if key == "POST /telemetry":
                self.telemetry_requests += 1
                if status < 300:
                    self.telemetry_accepted += 1
            elif key == "POST /predict":
                self.predict_requests += 1
                self.predict_latency_ms.append(latency_ms)
                if status != 200:
                    self.predict_errors += 1

    def record_inference(self, ms: float) -> None:
        with self._lock:
            self.inference_count += 1
            self.inference_sum_ms += ms
            self.inference_max_ms = max(self.inference_max_ms, ms)

    # --- lectura ---------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.predict_latency_ms)
            avg_lat = sum(lat) / len(lat) if lat else None
            avg_inf = self.inference_sum_ms / self.inference_count if self.inference_count else None

            def r(v: float | None) -> float | None:
                return None if v is None else round(v, 3)

            return {
                "started_at_utc": self.started_at.isoformat(),
                "uptime_seconds": round(time.monotonic() - self._t0, 1),
                "requests_received": self.total,
                "telemetry_requests": self.telemetry_requests,
                "telemetry_accepted": self.telemetry_accepted,
                "prediction_requests": self.predict_requests,
                "errors_total": self.errors_4xx + self.errors_5xx,
                "errors_4xx": self.errors_4xx,
                "errors_5xx": self.errors_5xx,
                "prediction_errors": self.predict_errors,
                "prediction_avg_inference_ms": r(avg_inf),
                "prediction_max_inference_ms": r(self.inference_max_ms if self.inference_count else None),
                "prediction_avg_latency_ms": r(avg_lat),
                "prediction_p50_latency_ms": r(_percentile(lat, 50)),
                "prediction_p95_latency_ms": r(_percentile(lat, 95)),
                "prediction_latency_window": len(lat),
                "requests_by_route": dict(sorted(self.by_route.items())),
                "status_codes": dict(sorted(self.by_status.items())),
            }


METRICS = Metrics()
