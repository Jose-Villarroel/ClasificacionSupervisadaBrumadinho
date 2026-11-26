"""Fase 3: HPO, selección, evaluación y modelo final.

La búsqueda usa solo los 64 polígonos de entrenamiento de results/split.csv.
El test de 24 polígonos queda congelado y se evalúa una sola vez, después de
seleccionar el modelo por Kappa de validación agrupada.

Uso:
    run.bat src\\05_tune_models.py
    run.bat src\\05_tune_models.py --models LR,SVM --folds 2
"""

import argparse
import hashlib
import json
import os
import platform
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             make_scorer, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     RandomizedSearchCV, StratifiedGroupKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import BANDS, CLASSES, FINAL_MODEL, MODEL_CARD, RESULTS, TSV

SEED = 42
BASE_SCORING = {
    "kappa": make_scorer(cohen_kappa_score),
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}
SCORE_METRICS = ("kappa", "f1_macro", "balanced_accuracy")

MODELS = {
    "LR": (
        "Regresión logística",
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        "grid",
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "DT": (
        "Árbol de decisión",
        DecisionTreeClassifier(random_state=SEED),
        "grid",
        {
            "clf__max_depth": [None, 4, 6, 8, 12],
            "clf__min_samples_leaf": [1, 2, 4, 8],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "RF": (
        "Random Forest",
        RandomForestClassifier(n_jobs=1, random_state=SEED),
        "random",
        {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [None, 8, 12, 16],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.5, None],
            "clf__class_weight": ["balanced"],
        },
    ),
    "SVM": (
        "Máquina de soporte vectorial",
        SVC(kernel="rbf", random_state=SEED),
        "grid",
        {
            "clf__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "clf__gamma": ["scale", 0.01, 0.03, 0.1, 1.0],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "ANN": (
        "Red neuronal (MLP)",
        MLPClassifier(max_iter=2000, random_state=SEED),
        "random",
        {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "clf__learning_rate_init": [1e-3, 1e-2],
        },
    ),
    "KNN": (
        "K vecinos más cercanos",
        KNeighborsClassifier(),
        "grid",
        {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    ),
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)

LABELS = sorted(CLASSES)
NAMES = [CLASSES[class_id] for class_id in LABELS]
N_BOOTSTRAP = 1000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(estimator))])


def load_training_data(folds):
    df = pd.read_csv(TSV, sep="\t")
    split = pd.read_csv(RESULTS / "split.csv")

    required = {"poly_id", "class_id", "set"}
    if not required.issubset(split.columns):
        raise ValueError(f"split.csv no tiene las columnas esperadas: {required}")

    train_ids = set(split.loc[split["set"] == "entrenamiento", "poly_id"])
    test_ids = set(split.loc[split["set"] == "prueba", "poly_id"])
    if not train_ids or not test_ids:
        raise ValueError("El split debe tener polígonos de entrenamiento y de prueba")
    if train_ids & test_ids:
        raise ValueError("Un polígono aparece en entrenamiento y prueba")
    if set(df["poly_id"]) != set(split["poly_id"]):
        raise ValueError("Los polígonos del TSV y de split.csv no coinciden")
    if len(test_ids) != 24:
        raise ValueError(f"Se esperaban 24 polígonos de prueba, hay {len(test_ids)}")
    if len(train_ids) != 64:
        raise ValueError(f"Se esperaban 64 polígonos de entrenamiento, hay {len(train_ids)}")

    train = df.loc[df["poly_id"].isin(train_ids)].reset_index(drop=True)
    if train["poly_id"].nunique() != len(train_ids):
        raise ValueError("Faltan píxeles de algún polígono de entrenamiento")

    X = train[BANDS].to_numpy(dtype=float) / 10000.0
    y = train["class_id"].to_numpy(dtype=int)
    groups = train["poly_id"].to_numpy()

    if len(np.unique(y)) != len(CLASSES):
        raise ValueError("No están representadas las seis clases en entrenamiento")
    if not 2 <= folds <= len(train_ids):
        raise ValueError(f"Número de folds inválido: {folds}")

    return df, split, X, y, groups, sorted(train_ids), sorted(test_ids)


def cv_results_frame(search, key, method):
    cv = search.cv_results_
    rows = []
    for index, params in enumerate(cv["params"]):
        row = {
            "model": key,
            "search": method,
            "candidate": index,
            "params": json.dumps(params, sort_keys=True, default=str),
            "mean_fit_time": float(cv["mean_fit_time"][index]),
            "rank_test_kappa": int(cv["rank_test_kappa"][index]),
        }
        for metric in SCORE_METRICS:
            row[f"mean_train_{metric}"] = float(cv[f"mean_train_{metric}"][index])
            row[f"std_train_{metric}"] = float(cv[f"std_train_{metric}"][index])
            row[f"mean_test_{metric}"] = float(cv[f"mean_test_{metric}"][index])
            row[f"std_test_{metric}"] = float(cv[f"std_test_{metric}"][index])
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(search, key, name, method, n_train_polys, n_train_pixels, elapsed):
    index = search.best_index_
    cv = search.cv_results_
    return {
        "model": key,
        "name": name,
        "search": method,
        "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
        "best_kappa": float(cv["mean_test_kappa"][index]),
        "best_kappa_std": float(cv["std_test_kappa"][index]),
        "best_f1_macro": float(cv["mean_test_f1_macro"][index]),
        "best_f1_macro_std": float(cv["std_test_f1_macro"][index]),
        "best_balanced_accuracy": float(cv["mean_test_balanced_accuracy"][index]),
        "best_train_kappa": float(cv["mean_train_kappa"][index]),
        "n_candidates": int(len(cv["params"])),
        "n_train_polys": int(n_train_polys),
        "n_train_pixels": int(n_train_pixels),
        "fit_seconds": float(elapsed),
    }


def build_search(method, estimator, space, cv, n_jobs, random_iter):
    search_kwargs = {
        "estimator": pipeline(estimator),
        "scoring": BASE_SCORING,
        "refit": "kappa",
        "cv": cv,
        "n_jobs": n_jobs,
        "return_train_score": True,
        "error_score": "raise",
    }
    if method == "random" and random_iter > 0:
        return RandomizedSearchCV(
            param_distributions=space,
            n_iter=random_iter,
            random_state=SEED,
            **search_kwargs,
        )
    return GridSearchCV(param_grid=space, **search_kwargs)


def metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }


def safe_metrics(y_true, y_pred):
    try:
        return metrics(y_true, y_pred)
    except ValueError:
        return {
            "oa": float(accuracy_score(y_true, y_pred)),
            "kappa": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
        }


def restore_params(params):
    restored = dict(params)
    hidden = restored.get("clf__hidden_layer_sizes")
    if hidden is not None:
        restored["clf__hidden_layer_sizes"] = tuple(hidden)
    return restored


def make_model(key, params):
    model = pipeline(MODELS[key][1])
    model.set_params(**restore_params(params))
    return model


def save_diagnostics(prefix, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(matrix, index=NAMES, columns=NAMES).to_csv(
        RESULTS / f"phase3_confusion_{prefix}.csv"
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    pd.DataFrame({
        "class_id": LABELS,
        "class_name": NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).to_csv(RESULTS / f"phase3_per_class_{prefix}.csv", index=False)


def make_bootstrap_samples(test_polygons, n_bootstrap):
    by_class = {
        int(class_id): frame["poly_id"].to_numpy()
        for class_id, frame in test_polygons.groupby("class_id")
    }
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(n_bootstrap):
        sample = [
            polygon_id
            for polygons in by_class.values()
            for polygon_id in rng.choice(polygons, size=len(polygons), replace=True)
        ]
        samples.append(sample)
    return samples


def bootstrap_summary(y_true, y_pred, pixels_by_polygon, samples):
    values = {metric: [] for metric in ("oa", "kappa", "f1_macro")}
    for sample in samples:
        indexes = np.concatenate([pixels_by_polygon[polygon_id] for polygon_id in sample])
        result = safe_metrics(y_true[indexes], y_pred[indexes])
        for metric in values:
            values[metric].append(result[metric])
    rows = []
    for metric, metric_values in values.items():
        metric_values = np.asarray(metric_values, dtype=float)
        low, high = np.nanpercentile(metric_values, [2.5, 97.5])
        rows.append({
            "metric": metric,
            "estimate": safe_metrics(y_true, y_pred)[metric],
            "ci_low": float(low),
            "ci_high": float(high),
            "n_bootstrap": len(samples),
        })
    return rows


def stability_row(key, params, X, y, groups):
    logo = LeaveOneGroupOut()
    oof = np.empty(len(y), dtype=int)
    fold_oa = []
    for train_index, test_index in logo.split(X, y, groups):
        model = make_model(key, params)
        model.fit(X[train_index], y[train_index])
        oof[test_index] = model.predict(X[test_index])
        fold_oa.append(float(accuracy_score(y[test_index], oof[test_index])))
    result = metrics(y, oof)
    return {
        "model": key,
        "oa_mean": float(np.mean(fold_oa)),
        "oa_se": float(np.std(fold_oa) / np.sqrt(len(fold_oa))),
        "kappa_oof": result["kappa"],
        "f1_oof": result["f1_macro"],
        "n_folds": int(len(fold_oa)),
    }


def select_hyperparameters(results, key):
    rows = results.loc[results["model"] == key].copy()
    best = rows.loc[rows["rank_test_kappa"] == 1].iloc[0]
    threshold = float(best["mean_test_kappa"] - best["std_test_kappa"])
    candidates = rows.loc[rows["mean_test_kappa"] >= threshold].copy()
    candidates["params_dict"] = candidates["params"].apply(json.loads)

    balanced = candidates.loc[
        candidates["params_dict"].apply(
            lambda params: params.get("clf__class_weight") == "balanced"
        )
    ]
    pool = balanced if not balanced.empty else candidates
    chosen = pool.sort_values(
        ["std_test_kappa", "mean_test_f1_macro", "mean_test_kappa"],
        ascending=[True, False, False],
    ).iloc[0]

    return {
        "family_best_params": json.loads(best["params"]),
        "params": json.loads(chosen["params"]),
        "family_best_kappa": float(best["mean_test_kappa"]),
        "family_best_std": float(best["std_test_kappa"]),
        "kappa": float(chosen["mean_test_kappa"]),
        "kappa_std": float(chosen["std_test_kappa"]),
        "f1_macro": float(chosen["mean_test_f1_macro"]),
        "one_se_threshold": threshold,
        "n_candidates_within_one_se": int(len(candidates)),
        "rule": (
            "Familia con mayor Kappa CV. Dentro de la familia, candidatos "
            "a una desviación estándar del mejor; se prefieren pesos "
            "balanceados y la menor desviación entre folds."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="HPO agrupado por polígono para Fase 3")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Modelos separados por coma: LR,DT,RF,SVM,ANN,KNN",
    )
    parser.add_argument("--folds", type=int, default=4, help="Folds agrupados")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Procesos paralelos")
    parser.add_argument(
        "--random-iter",
        type=int,
        default=30,
        help="Iteraciones para RF y ANN; 0 usa la malla completa",
    )
    args = parser.parse_args()

    requested = [item.strip().upper() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in requested if item not in MODELS]
    if unknown:
        raise ValueError(f"Modelos desconocidos: {unknown}")
    if not requested:
        raise ValueError("Debes indicar al menos un modelo")

    RESULTS.mkdir(exist_ok=True)
    df, split, X, y, groups, train_ids, test_ids = load_training_data(args.folds)
    cv = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=SEED
    )

    config = {
        "phase": 3,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "folds": args.folds,
        "n_jobs": args.n_jobs,
        "random_iter": args.random_iter,
        "metrics": {key: "Kappa" if key == "kappa" else key for key in SCORE_METRICS},
        "refit": "kappa",
        "n_train_polys": len(train_ids),
        "n_test_polys": len(test_ids),
        "n_train_pixels": int(len(y)),
        "n_test_pixels": int(len(df) - len(y)),
        "train_polys": train_ids,
        "test_polys": test_ids,
        "tsv_sha256": sha256(TSV),
        "split_sha256": sha256(RESULTS / "split.csv"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    (RESULTS / "phase3_hpo_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_results = []
    summary = []
    best_params = {}

    print(
        f"HPO Fase 3 | {len(train_ids)} polígonos / {len(y)} píxeles de entrenamiento | "
        f"test congelado: {len(test_ids)} polígonos"
    )

    for key in requested:
        name, estimator, method, space = MODELS[key]
        n_candidates = args.random_iter if method == "random" and args.random_iter > 0 else None
        search = build_search(
            method, estimator, space, cv, args.n_jobs, args.random_iter
        )
        start = time.perf_counter()
        search.fit(X, y, groups=groups)
        elapsed = time.perf_counter() - start

        frame = cv_results_frame(search, key, method)
        all_results.append(frame)
        summary.append(
            summary_row(
                search,
                key,
                name,
                method,
                len(train_ids),
                len(y),
                elapsed,
            )
        )
        best_params[key] = {
            "name": name,
            "search": method,
            "params": search.best_params_,
            "cv_kappa": float(search.best_score_),
        }
        if n_candidates is not None and len(frame) != n_candidates:
            print(f"{key}: se evaluaron {len(frame)} candidatos")
        print(
            f"{key:<4} Kappa={search.best_score_:.4f} | "
            f"F1={search.cv_results_['mean_test_f1_macro'][search.best_index_]:.4f} | "
            f"{len(frame)} candidatos | {elapsed:.1f}s"
        )

    results = pd.concat(all_results, ignore_index=True).sort_values(
        ["model", "rank_test_kappa", "candidate"]
    )
    summary_df = pd.DataFrame(summary).sort_values("best_kappa", ascending=False)

    selected_key = str(summary_df.iloc[0]["model"])
    selected_name = MODELS[selected_key][0]
    selection = select_hyperparameters(results, selected_key)
    best_params[selected_key] = {
        "name": selected_name,
        "search": MODELS[selected_key][2],
        "params": selection["params"],
        "cv_kappa": selection["kappa"],
        "cv_family_best_kappa": selection["family_best_kappa"],
        "selection_rule": selection["rule"],
    }

    results.to_csv(RESULTS / "phase3_hpo_results.csv", index=False)
    summary_df.to_csv(RESULTS / "phase3_cv_summary.csv", index=False)
    (RESULTS / "phase3_best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nMejores modelos por Kappa de validación agrupada:")
    print(summary_df[["model", "best_kappa", "best_f1_macro", "n_candidates"]].to_string(index=False))
    print(
        f"Selección robusta en {selected_key}: Kappa CV={selection['kappa']:.4f} "
        f"(mejor familia={selection['family_best_kappa']:.4f}, "
        f"umbral 1 EE={selection['one_se_threshold']:.4f})"
    )

    cv_rank = {
        str(model): rank
        for rank, model in enumerate(summary_df["model"], start=1)
    }
    cv_lookup = summary_df.set_index("model")

    test_mask = df["poly_id"].isin(test_ids).to_numpy()
    df_test = df.loc[test_mask].reset_index(drop=True)
    X_test = df_test[BANDS].to_numpy(dtype=float) / 10000.0
    y_test = df_test["class_id"].to_numpy(dtype=int)
    if len(df_test) != 472 or df_test["poly_id"].nunique() != 24:
        raise ValueError("El test congelado no tiene 24 polígonos / 472 píxeles")
    pixels_by_polygon = {
        polygon_id: np.flatnonzero(df_test["poly_id"].to_numpy() == polygon_id)
        for polygon_id in test_ids
    }
    test_polygons = split.loc[split["set"] == "prueba"].reset_index(drop=True)
    samples = make_bootstrap_samples(test_polygons, N_BOOTSTRAP)

    test_rows = []
    bootstrap_rows = []
    stability_rows = []

    print(f"\nEvaluación en el test congelado | modelo seleccionado por CV: {selected_key}")
    for key in requested:
        params = best_params[key]["params"]
        model = make_model(key, params)
        model.fit(X, y)
        prediction = model.predict(X_test)
        result = metrics(y_test, prediction)
        save_diagnostics(key, y_test, prediction)
        test_rows.append({
            "model": key,
            "name": MODELS[key][0],
            "selected_by_cv": key == selected_key,
            "cv_rank": int(cv_rank[key]),
            "cv_kappa": float(
                selection["kappa"] if key == selected_key
                else cv_lookup.loc[key, "best_kappa"]
            ),
            "cv_family_best_kappa": float(cv_lookup.loc[key, "best_kappa"]),
            **result,
            "n_test_polys": len(test_ids),
            "n_test_pixels": len(y_test),
        })
        for row in bootstrap_summary(y_test, prediction, pixels_by_polygon, samples):
            bootstrap_rows.append({"model": key, **row})
        stability_rows.append(stability_row(key, params, X, y, groups))
        print(
            f"{key:<4} OA={result['oa']:.4f} Kappa={result['kappa']:.4f} "
            f"F1={result['f1_macro']:.4f}"
        )

    test_df = pd.DataFrame(test_rows).sort_values("kappa", ascending=False)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    stability_df = pd.DataFrame(stability_rows)
    test_df.to_csv(RESULTS / "phase3_test_metrics.csv", index=False)
    bootstrap_df.to_csv(RESULTS / "phase3_bootstrap.csv", index=False)
    stability_df.to_csv(RESULTS / "phase3_stability.csv", index=False)

    all_X = df[BANDS].to_numpy(dtype=float) / 10000.0
    all_y = df["class_id"].to_numpy(dtype=int)
    final_model = make_model(selected_key, best_params[selected_key]["params"])
    final_model.fit(all_X, all_y)
    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "model_key": selected_key,
        "features": BANDS,
        "classes": LABELS,
        "params": best_params[selected_key]["params"],
    }
    joblib.dump(artifact, FINAL_MODEL, compress=3)

    selected_test = test_df.loc[test_df["model"] == selected_key].iloc[0].to_dict()
    selected_family_best = summary_df.loc[
        summary_df["model"] == selected_key
    ].iloc[0].to_dict()
    selected_bootstrap = {
        row["metric"]: {
            "estimate": row["estimate"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        }
        for row in bootstrap_rows
        if row["model"] == selected_key
    }
    selected_stability = stability_df.loc[
        stability_df["model"] == selected_key
    ].iloc[0].to_dict()

    model_card = {
        "phase": 3,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selected_model": selected_key,
        "selected_name": selected_name,
        "selection_rule": selection["rule"],
        "test_best_model": str(test_df.iloc[0]["model"]),
        "test_best_kappa": float(test_df.iloc[0]["kappa"]),
        "cv": selection,
        "cv_family_best": selected_family_best,
        "test": selected_test,
        "bootstrap": selected_bootstrap,
        "stability": selected_stability,
        "params": best_params[selected_key]["params"],
        "features": BANDS,
        "classes": {str(key): value for key, value in CLASSES.items()},
        "train": {
            "polygons": int(df["poly_id"].nunique()),
            "pixels": int(len(df)),
        },
        "cv_set": {
            "polygons": len(train_ids),
            "pixels": int(len(y)),
        },
        "test_set": {
            "polygons": len(test_ids),
            "pixels": int(len(y_test)),
        },
        "seed": SEED,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "hashes": {
            "tsv_sha256": sha256(TSV),
            "split_sha256": sha256(RESULTS / "split.csv"),
        },
        "artifact": str(FINAL_MODEL.relative_to(FINAL_MODEL.parents[1])),
    }
    MODEL_CARD.write_text(
        json.dumps(
            model_card,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )

    print("\nMétricas en test congelado:")
    print(
        test_df[["model", "oa", "kappa", "f1_macro", "selected_by_cv"]].to_string(
            index=False
        )
    )
    print(f"\nModelo final: {selected_key} ({selected_name})")
    print(f"Modelo: {FINAL_MODEL}")
    print(f"Ficha:  {MODEL_CARD}")
    print("Resultados en", RESULTS)


if __name__ == "__main__":
    main()
