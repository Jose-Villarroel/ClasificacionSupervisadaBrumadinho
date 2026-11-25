"""Fase 3: optimización de hiperparámetros agrupada por polígono.

La búsqueda usa solo los 64 polígonos de entrenamiento de results/split.csv.
El test de 24 polígonos queda congelado y no participa en la selección.

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

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import BANDS, CLASSES, RESULTS, TSV

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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


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

    results.to_csv(RESULTS / "phase3_hpo_results.csv", index=False)
    summary_df.to_csv(RESULTS / "phase3_cv_summary.csv", index=False)
    (RESULTS / "phase3_best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nMejores modelos por Kappa de validación agrupada:")
    print(summary_df[["model", "best_kappa", "best_f1_macro", "n_candidates"]].to_string(index=False))
    print("\nResultados en", RESULTS)


if __name__ == "__main__":
    main()
