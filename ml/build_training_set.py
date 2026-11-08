"""Genera varias flotas (una por semilla) y las une en data/training.csv con fleet_id.

Uso (desde la raíz del repo):
    python ml/build_training_set.py --fleets 8 --samples 3000

Cada semilla distinta es una flota distinta. El split train/test de train_model.py
se hace por fleet_id, así que el test siempre contiene flotas que el modelo no vio.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fleets", type=int, default=8)
    p.add_argument("--samples", type=int, default=3000, help="muestras por flota")
    p.add_argument("--robots", type=int, default=8, help="robots por flota")
    p.add_argument("--base-seed", type=int, default=1)
    p.add_argument("--output", default="data/training.csv")
    args = p.parse_args()

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(args.fleets):
        seed = args.base_seed + i
        raw = raw_dir / f"fleet_{seed}.csv"
        subprocess.run(
            [
                sys.executable, "simulator/generate_dataset.py",
                "--samples", str(args.samples),
                "--robots", str(args.robots),
                "--seed", str(seed),
                "--output", str(raw),
            ],
            check=True,
        )
        df = pd.read_csv(raw)
        df.insert(0, "fleet_id", f"F{seed}")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print(f"\nOK: {len(full)} filas, {args.fleets} flotas -> {out}")
    print(full.groupby("fleet_id")["status"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
