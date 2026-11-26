"""Fase 2 - Entrenamiento y evaluación inicial de los modelos.

Dos pistas con la misma partición por polígono:

  A. Oficial: entrena con los píxeles del TSV (12 bandas) y evalúa por píxel
     y por polígono (voto mayoritario y pureza).
  B. Experimental: entrena con la mediana de cada polígono más índices
     espectrales y evalúa por polígono.

La validación cruzada es LeaveOneGroupOut por polígono. Se reporta la OA
out-of-fold y el error estándar de la OA por pliegue, además de IC bootstrap
al 95 %. El entregable de la Fase 1 sigue siendo el TSV por píxel.

Uso:  run.bat src\\03_train_models.py
Salidas: results/*.csv y results/tsv_quality.json
"""
import json
import os
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import BANDS, CLASSES, INDICES, POLY_TSV, RESULTS, TSV

SEED = 42
MIN_TEST_POLYS = 4       # mínimo de polígonos de prueba por clase
TEST_FRAC = 0.25         # fracción de prueba cuando hay más polígonos
N_BOOTSTRAP = 1000
LABELS = sorted(CLASSES)  # 1..6
NAMES = [CLASSES[c] for c in LABELS]

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\..*")

MODELS = {
    "LR": ("Regresión logística",
           LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    "DT": ("Árbol de decisión",
           DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
    "RF": ("Random Forest",
           RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=1)),
    "SVM": ("Máquina de soporte vectorial",
            SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
    "ANN": ("Red neuronal (MLP)",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=SEED)),
    "KNN": ("K vecinos más cercanos",
            KNeighborsClassifier(n_neighbors=5)),
}


def pipeline(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def params(est):
    keys = ("max_iter", "class_weight", "n_estimators", "kernel", "C", "gamma",
            "hidden_layer_sizes", "n_neighbors", "max_depth", "criterion")
    return json.dumps({k: v for k, v in est.get_params().items() if k in keys},
                      default=str)


def split_by_polygon(df):
    """Por clase, una fracción de polígonos (mínimo 4) va a prueba."""
    rng = np.random.default_rng(SEED)
    polys = df.groupby("poly_id").agg(class_id=("class_id", "first"),
                                      class_name=("class_name", "first"),
                                      n_pixels=("class_id", "size")).reset_index()
    # Los polígonos nuevos (sufijo > 12) se usan solo para entrenamiento.
    # Así la prueba queda en los polígonos originales y la comparación
    # antes/después no depende de candidatos elegidos por su firma espectral.
    suffix = polys["poly_id"].str.extract(r"_(\d+)$")[0].astype(int)
    test = set()
    for _, g in polys.groupby("class_id"):
        ids = sorted(g.loc[suffix.loc[g.index] <= 12, "poly_id"])
        n_test = min(len(ids) - 1, max(MIN_TEST_POLYS, round(TEST_FRAC * len(ids))))
        test.update(rng.choice(ids, size=n_test, replace=False))
    polys["set"] = np.where(polys["poly_id"].isin(test), "prueba", "entrenamiento")
    return polys


def quality(df):
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "poligonos": int(df["poly_id"].nunique()),
        "valores_nulos": int(df.isna().sum().sum()),
        "filas_duplicadas": int(df.duplicated().sum()),
        "coordenadas_duplicadas": int(df.duplicated(["Latitude", "Longitude"]).sum()),
        "dn_min": int(df[BANDS].min().min()),
        "dn_max": int(df[BANDS].max().max()),
        "pixeles_por_clase": {CLASSES[k]: int(v) for k, v in
                              df["class_id"].value_counts().sort_index().items()},
    }


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
    }


def safe_metrics(y_true, y_pred):
    try:
        return metrics(y_true, y_pred)
    except ValueError:
        return {"oa": float(accuracy_score(y_true, y_pred)), "kappa": np.nan,
                "precision_macro": np.nan, "recall_macro": np.nan, "f1_macro": np.nan}


def save_confusion(prefix, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(
        RESULTS / f"confusion_{prefix}.csv")


def save_per_class(prefix, y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    pd.DataFrame({"class_id": LABELS, "class_name": NAMES, "precision": p,
                  "recall": r, "f1": f, "support": s}).to_csv(
        RESULTS / f"per_class_{prefix}.csv", index=False)


def aggregate_by_polygon(poly_ids, y_true, y_pred):
    """Voto mayoritario y pureza de las predicciones de cada polígono."""
    tmp = pd.DataFrame({"poly_id": np.asarray(poly_ids),
                        "y_true": np.asarray(y_true),
                        "y_pred": np.asarray(y_pred)})
    rows = []
    for pid, g in tmp.groupby("poly_id", sort=False):
        counts = np.bincount(g["y_pred"].to_numpy(dtype=int), minlength=max(LABELS) + 1)
        majority = int(np.argmax(counts))
        true = int(g["y_true"].iloc[0])
        rows.append({
            "poly_id": pid, "y_true": true, "y_pred": majority,
            "purity": float(counts[majority] / len(g)),
            "pixel_accuracy": float((g["y_pred"] == true).mean()),
            "n_pixels": int(len(g)),
        })
    return pd.DataFrame(rows)


def icc_table(df):
    """ICC de un factor (polígono) por banda y tamaño efectivo de muestra."""
    rows = []
    for band in BANDS:
        groups = [g[band].to_numpy(dtype=float) for _, g in df.groupby("poly_id")]
        k, n = len(groups), sum(len(g) for g in groups)
        m = n / k
        grand = np.concatenate(groups).mean()
        msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
        den = msb + (m - 1) * msw
        icc = (msb - msw) / den if den > 0 else 1.0
        rows.append({"band": band, "icc": float(icc), "n_pixels": int(n),
                     "n_polygons": int(k), "pixels_per_polygon_mean": float(m),
                     "n_eff": float(n / (1 + (m - 1) * icc))})
    return pd.DataFrame(rows)


def resample_polygons(rng, pids_by_class):
    return [pid for pids in pids_by_class.values()
            for pid in rng.choice(pids, size=len(pids), replace=True)]


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(TSV, sep="\t")
    poly = pd.read_csv(POLY_TSV, sep="\t")
    (RESULTS / "tsv_quality.json").write_text(
        json.dumps(quality(df), indent=2, ensure_ascii=False), encoding="utf-8")

    X = df[BANDS].to_numpy(dtype=float) / 10000.0
    y = df["class_id"].to_numpy()
    groups = df["poly_id"].to_numpy()

    polys = split_by_polygon(df)
    polys.to_csv(RESULTS / "split.csv", index=False)
    test_ids = set(polys.loc[polys["set"] == "prueba", "poly_id"])
    is_test = df["poly_id"].isin(test_ids).to_numpy()
    assert not set(groups[~is_test]) & set(groups[is_test]), "Un polígono quedó en ambos conjuntos"
    poly_test = poly["poly_id"].isin(test_ids).to_numpy()
    missing = set(polys["poly_id"]) - set(poly["poly_id"])
    assert not missing, f"Polígonos sin fila en {POLY_TSV.name}: {sorted(missing)}"

    XB = np.column_stack([poly[BANDS].to_numpy(dtype=float) / 10000.0,
                          poly[INDICES].to_numpy(dtype=float)])
    features_B = BANDS + INDICES
    yB = poly["class_id"].to_numpy()
    groupsB = poly["poly_id"].to_numpy()

    print(f"Entrenamiento: {(~is_test).sum()} píxeles / {len(set(groups[~is_test]))} polígonos | "
          f"Prueba: {is_test.sum()} píxeles / {len(test_ids)} polígonos")
    print(f"Pista B: {int((~poly_test).sum())} polígonos de entrenamiento / "
          f"{int(poly_test.sum())} de prueba, {len(features_B)} variables")

    logo = LeaveOneGroupOut()
    summary, polygon_rows, polytrain_rows, comparison = [], [], [], []
    cv_rows, bootstrap_rows = [], []
    importance_A, importance_B = {}, {}

    # Muestras bootstrap compartidas por modelo y pista (comparación pareada).
    test_polys = polys[polys["set"] == "prueba"]
    pids_by_class = {int(cid): g["poly_id"].to_numpy()
                     for cid, g in test_polys.groupby("class_id")}
    rng = np.random.default_rng(SEED)
    boot_samples = [resample_polygons(rng, pids_by_class) for _ in range(N_BOOTSTRAP)]

    df_test = df.loc[is_test].reset_index(drop=True)
    y_test = y[is_test]
    pix_by_pid = {pid: np.flatnonzero(df_test["poly_id"].to_numpy() == pid)
                  for pid in test_ids}

    for key, (name, est) in MODELS.items():
        # ---------- Pista A: píxeles (oficial) ----------
        model_A = pipeline(clone(est))
        t0 = time.perf_counter()
        model_A.fit(X[~is_test], y[~is_test])
        fit_A = time.perf_counter() - t0
        pred_A = model_A.predict(X[is_test])
        mA = metrics(y_test, pred_A)
        save_confusion(key, y_test, pred_A)
        save_per_class(key, y_test, pred_A)

        poly_A = aggregate_by_polygon(df_test["poly_id"].to_numpy(), y_test, pred_A)
        mpA = metrics(poly_A["y_true"], poly_A["y_pred"])
        mpA.update({"mean_purity": float(poly_A["purity"].mean()),
                    "mean_pixel_accuracy": float(poly_A["pixel_accuracy"].mean())})
        save_confusion(f"polygon_{key}", poly_A["y_true"], poly_A["y_pred"])
        save_per_class(f"polygon_{key}", poly_A["y_true"], poly_A["y_pred"])

        # Validación cruzada dejando un polígono fuera.
        oof_A = np.empty(len(y), dtype=int)
        fold_A = []
        for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
            m = pipeline(clone(est))
            m.fit(X[tr], y[tr])
            oof_A[te] = m.predict(X[te])
            fold_A.append(float(accuracy_score(y[te], oof_A[te])))
            cv_rows.append({"track": "pixel_A", "model": key, "fold": fold,
                            "group": groups[te][0], "n": int(len(te)),
                            "oa": fold_A[-1]})
        oofA = metrics(y, oof_A)
        poly_A_oof = aggregate_by_polygon(groups, y, oof_A)
        mpA_oof = metrics(poly_A_oof["y_true"], poly_A_oof["y_pred"])

        # ---------- Pista B: mediana por polígono (experimental) ----------
        model_B = pipeline(clone(est))
        t0 = time.perf_counter()
        model_B.fit(XB[~poly_test], yB[~poly_test])
        fit_B = time.perf_counter() - t0
        pred_B = model_B.predict(XB[poly_test])
        poly_B = pd.DataFrame({"poly_id": poly["poly_id"].to_numpy()[poly_test],
                               "y_true": yB[poly_test], "y_pred": pred_B})
        mB = metrics(poly_B["y_true"], poly_B["y_pred"])
        save_confusion(f"polytrain_{key}", poly_B["y_true"], poly_B["y_pred"])
        save_per_class(f"polytrain_{key}", poly_B["y_true"], poly_B["y_pred"])

        oof_B = np.empty(len(yB), dtype=int)
        fold_B = []
        for fold, (tr, te) in enumerate(logo.split(XB, yB, groupsB)):
            m = pipeline(clone(est))
            m.fit(XB[tr], yB[tr])
            oof_B[te] = m.predict(XB[te])
            fold_B.append(float(accuracy_score(yB[te], oof_B[te])))
            cv_rows.append({"track": "polygon_B", "model": key, "fold": fold,
                            "group": groupsB[te][0], "n": int(len(te)),
                            "oa": fold_B[-1]})
        oofB = metrics(yB, oof_B)

        # ---------- Filas de resumen ----------
        summary.append({
            "model": key, "name": name, **mA,
            "cv_oa_mean": oofA["oa"],
            "cv_oa_std": float(np.std(fold_A) / np.sqrt(len(fold_A))),
            "cv_kappa_mean": oofA["kappa"], "cv_kappa_std": np.nan,
            "cv_oof_oa": oofA["oa"], "cv_oof_kappa": oofA["kappa"],
            "cv_oof_f1_macro": oofA["f1_macro"],
            "n_train_pixels": int((~is_test).sum()), "n_test_pixels": int(is_test.sum()),
            "n_train_polys": int(len(set(groups[~is_test]))),
            "n_test_polys": int(len(test_ids)),
            "fit_seconds": fit_A, "params": params(est),
        })
        polygon_rows.append({
            "model": key, "name": name, **mpA,
            "cv_oof_oa": mpA_oof["oa"], "cv_oof_kappa": mpA_oof["kappa"],
            "cv_oof_f1_macro": mpA_oof["f1_macro"],
            "n_test_polygons": int(len(poly_A)),
        })
        polytrain_rows.append({
            "model": key, "name": name, **mB,
            "cv_oa_mean": oofB["oa"],
            "cv_oa_std": float(np.std(fold_B) / np.sqrt(len(fold_B))),
            "cv_kappa_mean": oofB["kappa"], "cv_kappa_std": np.nan,
            "cv_oof_oa": oofB["oa"], "cv_oof_kappa": oofB["kappa"],
            "cv_oof_f1_macro": oofB["f1_macro"],
            "n_train_polys": int((~poly_test).sum()), "n_test_polys": int(poly_test.sum()),
            "n_features": int(len(features_B)),
            "fit_seconds": fit_B, "params": params(est),
        })
        comparison.append({
            "model": key, "name": name,
            "oa_pixel": mA["oa"], "kappa_pixel": mA["kappa"], "f1_pixel": mA["f1_macro"],
            "oa_polygon_A": mpA["oa"], "kappa_polygon_A": mpA["kappa"],
            "f1_polygon_A": mpA["f1_macro"], "purity_A": mpA["mean_purity"],
            "oa_polygon_B": mB["oa"], "kappa_polygon_B": mB["kappa"],
            "f1_polygon_B": mB["f1_macro"],
        })

        # ---------- IC bootstrap ----------
        row_A = {pid: i for i, pid in enumerate(poly_A["poly_id"])}
        row_B = {pid: i for i, pid in enumerate(poly_B["poly_id"])}
        yt_A = poly_A["y_true"].to_numpy()
        yp_A = poly_A["y_pred"].to_numpy()
        yt_B = poly_B["y_true"].to_numpy()
        yp_B = poly_B["y_pred"].to_numpy()
        tracks = [
            ("pixel_A", mA,
             lambda s: (y_test[np.concatenate([pix_by_pid[p] for p in s])],
                        pred_A[np.concatenate([pix_by_pid[p] for p in s])])),
            ("polygon_A", mpA,
             lambda s: (yt_A[[row_A[p] for p in s]], yp_A[[row_A[p] for p in s]])),
            ("polygon_B", mB,
             lambda s: (yt_B[[row_B[p] for p in s]], yp_B[[row_B[p] for p in s]])),
        ]
        for track, estimate, sampler in tracks:
            for metric in ("oa", "kappa", "f1_macro"):
                values = []
                for sample in boot_samples:
                    yt, yp = sampler(sample)
                    values.append(safe_metrics(yt, yp)[metric])
                values = np.asarray(values, dtype=float)
                lo, hi = np.nanpercentile(values, [2.5, 97.5])
                bootstrap_rows.append({
                    "track": track, "model": key, "metric": metric,
                    "estimate": estimate[metric], "ci_low": lo, "ci_high": hi,
                })

        # ---------- Importancia de variables ----------
        for tag, model, names in (("A", model_A, BANDS), ("B", model_B, features_B)):
            clf = model.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                (importance_A if tag == "A" else importance_B)[key] = clf.feature_importances_

        print(f"{key:<4} A-pixel OA={mA['oa']:.3f} K={mA['kappa']:.3f} | "
              f"A-pol OA={mpA['oa']:.3f} K={mpA['kappa']:.3f} | "
              f"B-pol OA={mB['oa']:.3f} K={mB['kappa']:.3f} | "
              f"LOO-A={np.mean(fold_A):.3f}")

    pd.DataFrame(summary).to_csv(RESULTS / "metrics_summary.csv", index=False)
    pd.DataFrame(polygon_rows).to_csv(RESULTS / "metrics_polygon.csv", index=False)
    pd.DataFrame(polytrain_rows).to_csv(RESULTS / "metrics_polytrain.csv", index=False)
    pd.DataFrame(comparison).to_csv(RESULTS / "pixel_vs_polygon.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(RESULTS / "cv_folds.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS / "bootstrap_ci.csv", index=False)
    pd.DataFrame(importance_A, index=BANDS).rename_axis("band").to_csv(
        RESULTS / "feature_importance.csv")
    if importance_B:
        pd.DataFrame(importance_B, index=features_B).rename_axis("feature").to_csv(
            RESULTS / "feature_importance_polytrain.csv")

    icc = icc_table(df)
    icc.to_csv(RESULTS / "icc.csv", index=False)
    print(f"ICC: {icc['icc'].min():.3f}-{icc['icc'].max():.3f} | "
          f"n_eff: {icc['n_eff'].min():.1f}-{icc['n_eff'].max():.1f}")
    print("Resultados en", RESULTS)


if __name__ == "__main__":
    main()
