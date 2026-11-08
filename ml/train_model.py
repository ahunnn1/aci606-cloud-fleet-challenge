"""Ejercicio 4 - pipeline de entrenamiento reproducible.

Uso (desde la raíz del repo):
    python ml/build_training_set.py --fleets 8 --samples 3000
    python ml/train_model.py

Qué hace:
  1. EDA: distribución de clases y resumen de variables por clase.
  2. Split train/test por fleet_id (GroupShuffleSplit): el test son flotas no vistas.
  3. Ablación de variables (leave-one-fleet-out, solo sobre train).
  4. Compara clasificadores básicos (regresión logística, árbol de decisión, k-NN) con validación cruzada agrupada (solo sobre train).
     Elige por F1 macro en CV; si hay empate técnico (<= 0.005) gana el más rápido.
  5. Evalúa en test (matriz de confusión, F1 macro, línea base, latencia).
  6. Reentrena con todas las flotas y guarda models/model.joblib (+ model_meta.json).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/training.csv")
MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/model_meta.json")
REPORTS = Path("reports")

# Fuera a propósito:
#  - robot_id, timestamp, x, y: identifican una flota/lugar concreto, no generalizan.
#  - obstacle_distance: en el simulador se genera independiente de la salud del robot
#    (core.py), por lo que no aporta información sobre el estado.
#  - battery: baja con el tiempo, igual que la salud; correlaciona con el estado solo
#    por el paso del tiempo (variable confusora) y la ablación muestra que no mejora el F1.
# La ablación de abajo deja esta decisión documentada con datos.
FEATURES = ["speed", "motor_temp", "vibration", "motor_current"]

ABLATION_SETS = {
    "baseline_6_variables": ["speed", "battery", "motor_temp", "vibration", "motor_current", "obstacle_distance"],
    "sin_obstacle_distance": ["speed", "battery", "motor_temp", "vibration", "motor_current"],
    "final_4_variables": FEATURES,
    "solo_vibration_y_temp": ["motor_temp", "vibration"],
    "solo_vibration": ["vibration"],
}
TIE_TOL = 0.005  # diferencia de F1 macro en CV que consideramos empate técnico
TARGET = "status"
CLASSES = ["NORMAL", "WARNING", "CRITICAL"]
SEED = 42


def candidates(extra: bool = False) -> dict[str, Pipeline]:
    """Modelos básicos por defecto. `extra=True` agrega modelos de ensamble (opcional)."""
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "decision_tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, class_weight="balanced",
                random_state=SEED)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),  # k-NN usa distancias: hay que estandarizar
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
    }
    if extra:
        models["random_forest"] = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                random_state=SEED, n_jobs=1)),
        ])
        models["hist_gradient_boosting"] = Pipeline([
            ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=SEED)),
        ])
    return models


def load_data() -> tuple[pd.DataFrame, str]:
    if not DATA.exists():
        raise SystemExit(
            "No existe data/training.csv. Ejecuta: python ml/build_training_set.py"
        )
    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas en el CSV: {missing}")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    if "fleet_id" in df.columns:
        group_col = "fleet_id"
    else:
        print("AVISO: el CSV no tiene fleet_id; se agrupa por robot_id (validación más débil).")
        group_col = "robot_id"
    if df[group_col].nunique() < 4:
        raise SystemExit(
            f"Hay solo {df[group_col].nunique()} grupos en '{group_col}'. "
            "Genera más flotas con ml/build_training_set.py --fleets 8"
        )
    return df, group_col


def eda(df: pd.DataFrame, group_col: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    dist = df[TARGET].value_counts().reindex(CLASSES).fillna(0).astype(int)
    pct = (dist / dist.sum() * 100).round(2)
    table = pd.DataFrame({"count": dist, "percent": pct})
    print("\n== Distribución de clases ==")
    print(table)
    table.to_csv(REPORTS / "class_distribution.csv")

    print(f"\n== Grupos ({group_col}): {df[group_col].nunique()} ==")
    print(df.groupby(group_col)[TARGET].value_counts().unstack(fill_value=0))

    summary = df.groupby(TARGET)[FEATURES].agg(["mean", "std"]).round(3)
    print("\n== Media/desv. de cada variable por clase ==")
    print(summary.T)
    summary.to_csv(REPORTS / "feature_summary_by_class.csv")

    try:  # gráficos opcionales: solo si matplotlib está instalado
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = dist.plot(kind="bar", color=["#4c9f70", "#e0a030", "#c0392b"])
        ax.set_title("Distribución de clases")
        ax.set_ylabel("muestras")
        plt.tight_layout()
        plt.savefig(REPORTS / "class_distribution.png", dpi=120)
        plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, feat in zip(axes.ravel(), FEATURES):
            data = [df.loc[df[TARGET] == c, feat] for c in CLASSES]
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
            ax.set_title(feat)
        plt.tight_layout()
        plt.savefig(REPORTS / "features_by_class.png", dpi=120)
        plt.close()
    except Exception as exc:  # los gráficos nunca deben romper el entrenamiento
        print(f"(se omiten los gráficos del EDA: {exc})")


def ablation(df_tr: pd.DataFrame, group_col: str) -> None:
    """F1 macro (media por flota, leave-one-fleet-out) con distintos subconjuntos de variables."""
    print("\n== Ablación de variables (logreg, leave-one-fleet-out sobre train) ==")
    rows = []
    for name, feats in ABLATION_SETS.items():
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        pred = cross_val_predict(pipe, df_tr[feats], df_tr[TARGET],
                                 groups=df_tr[group_col], cv=LeaveOneGroupOut())
        per_fleet = [
            f1_score(df_tr[TARGET][df_tr[group_col] == g], pred[(df_tr[group_col] == g).to_numpy()],
                     average="macro", labels=CLASSES, zero_division=0)
            for g in sorted(df_tr[group_col].unique())
        ]
        rows.append({"variables": name, "n_features": len(feats),
                     "f1_macro_mean": round(float(np.mean(per_fleet)), 4),
                     "f1_macro_min": round(float(np.min(per_fleet)), 4)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(REPORTS / "feature_ablation.csv", index=False)


def single_row_latency_ms(model: Pipeline, X: pd.DataFrame, n: int = 200) -> float:
    """Mediana de latencia de predict_proba con una sola fila (como en /predict)."""
    times = []
    for i in range(n):
        row = X.iloc[[i % len(X)]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def plot_confusion(cm: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), CLASSES, rotation=30)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(REPORTS / "confusion_matrix.png", dpi=120)
        plt.close()
    except Exception as exc:
        print(f"(se omite el gráfico de la matriz de confusión: {exc})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="model_version que reportará /predict")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--extra-models", action="store_true",
                    help="agrega Random Forest y Gradient Boosting a la comparación")
    args = ap.parse_args()

    df, group_col = load_data()
    eda(df, group_col)

    X, y, groups = df[FEATURES], df[TARGET], df[group_col]

    # --- Split train/test por flota -------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, y_tr, g_tr = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
    train_groups = sorted(g_tr.unique())
    test_groups = sorted(groups.iloc[te_idx].unique())
    print(f"\n== Split por {group_col} ==")
    print(f"train: {len(X_tr)} filas, grupos {train_groups}")
    print(f"test : {len(X_te)} filas, grupos {test_groups}")
    if set(y_te.unique()) != set(CLASSES):
        print("AVISO: el test no contiene las 3 clases; prueba otra semilla o más flotas.")

    ablation(df.iloc[tr_idx], group_col)

    # --- Comparación con CV agrupada (solo train) ----------------------------------
    n_splits = min(4, len(train_groups))
    cv = GroupKFold(n_splits=n_splits)
    models = candidates(args.extra_models)
    results = {}
    print(f"\n== Comparación (GroupKFold={n_splits} sobre train, métrica F1 macro) ==")
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_tr, y_tr, groups=g_tr, cv=cv, scoring="f1_macro")
        results[name] = {"cv_f1_macro_mean": float(scores.mean()), "cv_f1_macro_std": float(scores.std())}
        print(f"{name:<24} F1 macro = {scores.mean():.4f} ± {scores.std():.4f}")

    # --- Evaluación en flotas de test ----------------------------------------------
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    baseline_f1 = f1_score(y_te, baseline.predict(X_te), average="macro", labels=CLASSES, zero_division=0)

    print("\n== Test (flotas no vistas) ==")
    for name, pipe in models.items():
        fitted = pipe.fit(X_tr, y_tr)
        pred = fitted.predict(X_te)
        results[name]["test_f1_macro"] = float(f1_score(y_te, pred, average="macro", labels=CLASSES, zero_division=0))
        results[name]["latency_ms_single_row"] = round(single_row_latency_ms(fitted, X_te), 3)
        print(f"{name:<24} F1 macro test = {results[name]['test_f1_macro']:.4f} | "
              f"latencia 1 fila ≈ {results[name]['latency_ms_single_row']} ms")
    print(f"{'baseline (mayoritaria)':<24} F1 macro test = {baseline_f1:.4f}")

    # Selección: SOLO con CV (train) y latencia; el test no interviene en la decisión.
    best_cv = max(r["cv_f1_macro_mean"] for r in results.values())
    eligible = [n for n, r in results.items() if r["cv_f1_macro_mean"] >= best_cv - TIE_TOL]
    best_name = min(eligible, key=lambda n: results[n]["latency_ms_single_row"])
    print(f"\nCandidatos en empate técnico (CV >= mejor - {TIE_TOL}): {eligible}")
    print(f"Modelo elegido: {best_name} (el más rápido entre los empatados)")

    best = models[best_name]  # ya ajustado sobre train en el bucle anterior
    pred = best.predict(X_te)
    cm = confusion_matrix(y_te, pred, labels=CLASSES)
    print(f"\n== Modelo elegido: {best_name} ==")
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    print("\n" + classification_report(y_te, pred, labels=CLASSES, zero_division=0))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(REPORTS / "confusion_matrix.csv")
    plot_confusion(cm)

    # --- Reentrenar con todo y guardar ---------------------------------------------
    final = candidates(args.extra_models)[best_name].fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    meta = {
        "model_version": args.version,
        "model_name": best_name,
        "features": FEATURES,
        "classes": list(final.classes_),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "random_state": SEED,
        "group_column": group_col,
        "n_rows": int(len(df)),
        "n_groups": int(groups.nunique()),
        "test_groups": test_groups,
        "baseline_f1_macro_test": float(baseline_f1),
        "candidates": results,
        "chosen_test_f1_macro": results[best_name]["test_f1_macro"],
        "confusion_matrix_test": cm.tolist(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nModelo guardado en {MODEL_PATH} (versión {args.version})")
    print(f"Metadatos en {META_PATH}; evidencia en {REPORTS}/")


if __name__ == "__main__":
    main()
