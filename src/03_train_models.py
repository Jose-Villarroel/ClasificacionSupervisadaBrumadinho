"""Fase 2 - Entrenamiento y evaluación inicial de los modelos.

Lee data/training_dataset.tsv, separa entrenamiento y prueba POR POLÍGONO
(8 polígonos de entrenamiento y 4 de prueba por clase), entrena los cinco
modelos del enunciado (más Random Forest como bonus) con parámetros iniciales
y calcula matriz de confusión, exactitud global (OA), Kappa, precisión,
recall y F1. Además estima la OA con validación cruzada agrupada por polígono.

Uso:  run_ml.bat src\\03_train_models.py
Salidas: results/*.csv y results/tsv_quality.json
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             make_scorer, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import BANDS, CLASSES, RESULTS, TSV

SEED = 42
N_TEST_POLYS = 4          # polígonos de prueba por clase (de 12)
LABELS = sorted(CLASSES)  # 1..6

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Nombre corto -> (nombre largo, estimador con parámetros iniciales)
MODELS = {
    "LR": ("Regresión logística",
           LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    "DT": ("Árbol de decisión",
           DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
    "RF": ("Random Forest",
           RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)),
    "SVM": ("Máquina de soporte vectorial",
            SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
    "ANN": ("Red neuronal (MLP)",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=SEED)),
    "KNN": ("K vecinos más cercanos",
            KNeighborsClassifier(n_neighbors=5)),
}


def pipeline(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def split_by_polygon(df):
    """Por cada clase, N_TEST_POLYS polígonos al azar (semilla fija) van a prueba."""
    rng = np.random.default_rng(SEED)
    polys = df.groupby("poly_id").agg(class_id=("class_id", "first"),
                                      class_name=("class_name", "first"),
                                      n_pixels=("class_id", "size")).reset_index()
    test = set()
    for _, g in polys.groupby("class_id"):
        ids = sorted(g["poly_id"])
        test.update(rng.choice(ids, size=N_TEST_POLYS, replace=False))
    polys["set"] = np.where(polys["poly_id"].isin(test), "prueba", "entrenamiento")
    return polys


def quality(df):
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "valores_nulos": int(df.isna().sum().sum()),
        "filas_duplicadas": int(df.duplicated().sum()),
        "coordenadas_duplicadas": int(df.duplicated(["Latitude", "Longitude"]).sum()),
        "dn_min": int(df[BANDS].min().min()),
        "dn_max": int(df[BANDS].max().max()),
        "pixeles_por_clase": {CLASSES[k]: int(v) for k, v in df["class_id"].value_counts().sort_index().items()},
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(TSV, sep="\t")
    (RESULTS / "tsv_quality.json").write_text(
        json.dumps(quality(df), indent=2, ensure_ascii=False), encoding="utf-8")

    X = df[BANDS].to_numpy(dtype=float) / 10000.0   # reflectancia
    y = df["class_id"].to_numpy()
    groups = df["poly_id"].to_numpy()

    polys = split_by_polygon(df)
    polys.to_csv(RESULTS / "split.csv", index=False)
    test_ids = set(polys.loc[polys["set"] == "prueba", "poly_id"])
    is_test = df["poly_id"].isin(test_ids).to_numpy()
    assert not set(groups[~is_test]) & set(groups[is_test]), "Un polígono quedó en ambos conjuntos"
    print(f"Entrenamiento: {(~is_test).sum()} píxeles / {len(set(groups[~is_test]))} polígonos | "
          f"Prueba: {is_test.sum()} píxeles / {len(test_ids)} polígonos")

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    scoring = {"oa": "accuracy", "kappa": make_scorer(cohen_kappa_score)}

    rows, importances = [], {}
    for key, (name, est) in MODELS.items():
        model = pipeline(est)
        t0 = time.perf_counter()
        model.fit(X[~is_test], y[~is_test])
        fit_s = time.perf_counter() - t0
        pred = model.predict(X[is_test])
        true = y[is_test]

        cm = confusion_matrix(true, pred, labels=LABELS)
        names = [CLASSES[c] for c in LABELS]
        pd.DataFrame(cm, index=names, columns=names).to_csv(RESULTS / f"confusion_{key}.csv")

        p, r, f, s = precision_recall_fscore_support(true, pred, labels=LABELS, zero_division=0)
        pd.DataFrame({"class_id": LABELS, "class_name": names, "precision": p,
                      "recall": r, "f1": f, "support": s}).to_csv(
            RESULTS / f"per_class_{key}.csv", index=False)

        cvres = cross_validate(pipeline(est), X, y, groups=groups, cv=cv, scoring=scoring)

        rows.append({
            "model": key, "name": name,
            "oa": accuracy_score(true, pred),
            "kappa": cohen_kappa_score(true, pred),
            "precision_macro": p.mean(), "recall_macro": r.mean(), "f1_macro": f.mean(),
            "cv_oa_mean": cvres["test_oa"].mean(), "cv_oa_std": cvres["test_oa"].std(),
            "cv_kappa_mean": cvres["test_kappa"].mean(), "cv_kappa_std": cvres["test_kappa"].std(),
            "fit_seconds": fit_s,
            "params": json.dumps({k: v for k, v in est.get_params().items()
                                  if k in ("max_iter", "class_weight", "n_estimators", "kernel",
                                           "C", "gamma", "hidden_layer_sizes", "n_neighbors",
                                           "max_depth", "criterion")}, default=str),
        })
        if hasattr(model.named_steps["clf"], "feature_importances_"):
            importances[key] = model.named_steps["clf"].feature_importances_
        print(f"{key:<4} OA={rows[-1]['oa']:.3f}  Kappa={rows[-1]['kappa']:.3f}  "
              f"F1={rows[-1]['f1_macro']:.3f}  CV-OA={rows[-1]['cv_oa_mean']:.3f}±{rows[-1]['cv_oa_std']:.3f}")

    pd.DataFrame(rows).to_csv(RESULTS / "metrics_summary.csv", index=False)
    pd.DataFrame(importances, index=BANDS).rename_axis("band").to_csv(
        RESULTS / "feature_importance.csv")
    print("Resultados en", RESULTS)


if __name__ == "__main__":
    main()
