"""Smoke test del servicio.

    python tests/smoke_test.py --api http://127.0.0.1:8000
    python tests/smoke_test.py --api https://tu-servicio.onrender.com --final

Con --final se exige /predict completo (contrato, validación 4xx y concurrencia).
Sin --final, /predict puede responder 501 (starter) y no se valida.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

STATUSES = {"NORMAL", "WARNING", "CRITICAL"}
CONTRACT_KEYS = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def sample(**overrides) -> dict:
    base = {
        "robot_id": "TEST-01", "timestamp": time.time(), "x": 1.2, "y": 3.4,
        "speed": 0.72, "battery": 81.0, "motor_temp": 49.0, "vibration": 0.42,
        "motor_current": 3.8, "obstacle_distance": 1.3,
    }
    base.update(overrides)
    return base


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"{'OK  ' if ok else 'FAIL'}  {name:<46} {detail}")
        if not ok:
            self.failed += 1


def wake_up(base: str) -> None:
    """Render duerme el servicio tras 15 min sin tráfico: el primer request puede tardar ~1 min."""
    for attempt in range(6):
        try:
            if requests.get(base + "/health", timeout=30).ok:
                return
        except requests.RequestException:
            pass
        print(f"  esperando al servicio... ({attempt + 1}/6)")
        time.sleep(5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--final", action="store_true", help="exigir /predict y pruebas de robustez")
    args = p.parse_args()
    base = args.api.rstrip("/")
    r = Report()

    wake_up(base)

    # --- rutas básicas ---------------------------------------------------------------
    resp = requests.get(base + "/health", timeout=10)
    r.check("GET /health", resp.ok, f"-> {resp.status_code}")
    resp = requests.post(base + "/telemetry", json=sample(), timeout=10)
    r.check("POST /telemetry", resp.ok, f"-> {resp.status_code}")
    resp = requests.get(base + "/robots", timeout=10)
    r.check("GET /robots", resp.ok, f"-> {resp.status_code}")
    resp = requests.get(base + "/robots/TEST-01/history", timeout=10)
    r.check("GET /robots/TEST-01/history", resp.ok and len(resp.json()) >= 1, f"-> {resp.status_code}")
    resp = requests.get(base + "/metrics", timeout=10)
    r.check("GET /metrics", resp.ok, f"-> {resp.status_code}")
    resp = requests.get(base + "/dashboard", timeout=10)
    r.check("GET /dashboard", resp.ok, f"-> {resp.status_code}")

    # --- /predict ---------------------------------------------------------------------
    resp = requests.post(base + "/predict", json=sample(), timeout=10)
    if not args.final and resp.status_code == 501:
        r.check("POST /predict (starter, aún no implementado)", True, "-> 501")
    else:
        body = resp.json() if resp.ok else {}
        r.check("POST /predict responde 200", resp.ok, f"-> {resp.status_code}")
        if resp.ok:
            r.check("  claves del contrato", set(body) == CONTRACT_KEYS, str(sorted(body)))
            r.check("  robot_id devuelto igual al enviado", body.get("robot_id") == "TEST-01")
            r.check("  predicted_status válido", body.get("predicted_status") in STATUSES,
                    str(body.get("predicted_status")))
            conf = body.get("confidence")
            r.check("  confidence en [0, 1]", isinstance(conf, (int, float)) and 0 <= conf <= 1, str(conf))
            r.check("  model_version es texto", isinstance(body.get("model_version"), str),
                    str(body.get("model_version")))
            r.check("  inference_ms numérico >= 0",
                    isinstance(body.get("inference_ms"), (int, float)) and body["inference_ms"] >= 0,
                    str(body.get("inference_ms")))

    if args.final:
        # --- robustez: datos inválidos => 4xx (nunca 5xx) y el servicio sigue vivo ------
        headers = {"content-type": "application/json"}
        valid = '{"robot_id":"X","timestamp":1,"x":1,"y":1,"speed":1,"battery":50,"motor_temp":50,' \
                '"vibration":1,"motor_current":3,"obstacle_distance":1}'
        bad = {
            "campo faltante": {"json": {k: v for k, v in sample().items() if k != "vibration"}},
            "battery fuera de rango (150)": {"json": sample(battery=150)},
            "speed negativa": {"json": sample(speed=-1)},
            "texto en campo numérico": {"json": sample(motor_temp="abc")},
            "valor nulo": {"json": sample(vibration=None)},
            "NaN": {"data": valid.replace('"vibration":1', '"vibration":NaN'), "headers": headers},
            "Infinity": {"data": valid.replace('"motor_temp":50', '"motor_temp":Infinity'), "headers": headers},
            "JSON mal formado": {"data": "{no es json", "headers": headers},
            "cuerpo vacío": {"data": "", "headers": headers},
        }
        for route in ("/predict", "/telemetry"):
            for label, kwargs in bad.items():
                resp = requests.post(base + route, timeout=10, **kwargs)
                r.check(f"{route} {label} => 4xx", 400 <= resp.status_code < 500, f"-> {resp.status_code}")
        resp = requests.get(base + "/health", timeout=10)
        r.check("servicio activo tras datos inválidos", resp.ok, f"-> {resp.status_code}")

        # --- carga: solicitudes concurrentes a /predict ---------------------------------
        def call(_: int) -> tuple[int, float]:
            t0 = time.perf_counter()
            code = requests.post(base + "/predict", json=sample(), timeout=30).status_code
            return code, (time.perf_counter() - t0) * 1000

        n, workers = 200, 20
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(call, range(n)))
        codes = [c for c, _ in results]
        lat = sorted(ms for _, ms in results)
        p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95) - 1]
        r.check(f"{n} /predict concurrentes ({workers} hilos) sin errores", all(c == 200 for c in codes),
                f"errores={sum(c != 200 for c in codes)}")
        print(f"      latencia medida desde el cliente: p50={p50:.0f} ms, p95={p95:.0f} ms")

    print("\nRESULTADO:", "TODO OK" if r.failed == 0 else f"{r.failed} verificación(es) fallaron")
    raise SystemExit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
