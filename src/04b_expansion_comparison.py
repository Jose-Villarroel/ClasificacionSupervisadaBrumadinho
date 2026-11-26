"""Fase 2: regenera la comparación entre 72 y 88 polígonos.

Usa la base congelada de data/baseline_72 y la versión actual del dataset.
Produce los dos CSV de la comparación y las tablas LaTeX de report_fase2 y
report_completo. El entrenamiento de cada escenario excluye los polígonos que
se evalúan para no introducir fuga.

Uso:
    run.bat src\\04b_expansion_comparison.py
"""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import BANDS, CLASSES, DATA, RESULTS, TSV

ROOT = Path(__file__).resolve().parents[1]
BASE = DATA / "baseline_72"
BASE_TSV = BASE / "training_dataset.tsv"
BASE_SPLIT = BASE / "split.csv"
BASE_METRICS = BASE / "metrics_summary.csv"
BASE_PER_CLASS = {
    "SVM": BASE / "per_class_SVM.csv",
    "ANN": BASE / "per_class_ANN.csv",
}
TAB2 = ROOT / "report_fase2" / "tables"
TAB_FULL = ROOT / "report_completo" / "tables"
SEED = 42
LABELS = sorted(CLASSES)
NAMES = [CLASSES[class_id] for class_id in LABELS]

MODELS = {
    "LR": (
        "Regresión logística",
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=SEED),
    ),
    "DT": (
        "Árbol de decisión",
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    ),
    "RF": (
        "Random Forest",
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=1, random_state=SEED),
    ),
    "SVM": (
        "Máquina de soporte vectorial",
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            random_state=SEED),
    ),
    "ANN": (
        "Red neuronal (MLP)",
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                      random_state=SEED),
    ),
    "KNN": (
        "K vecinos más cercanos",
        KNeighborsClassifier(n_neighbors=5),
    ),
}


def pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(estimator))])


def read_split(path):
    split = pd.read_csv(path)
    train = set(split.loc[split["set"] == "entrenamiento", "poly_id"])
    test = set(split.loc[split["set"] == "prueba", "poly_id"])
    if train & test:
        raise ValueError(f"Polígonos compartidos entre train y test en {path}")
    return split, train, test


def load_data():
    old = pd.read_csv(BASE_TSV, sep="\t")
    new = pd.read_csv(TSV, sep="\t")
    old_split, old_train, old_test = read_split(BASE_SPLIT)
    new_split, new_train, new_test = read_split(RESULTS / "split.csv")

    if old["poly_id"].nunique() != 72:
        raise ValueError("La base de 72 polígonos no tiene 72 polígonos")
    if new["poly_id"].nunique() != 88:
        raise ValueError("El dataset actual no tiene 88 polígonos")
    if len(old_train) != 48 or len(old_test) != 24:
        raise ValueError("La división antigua debe tener 48/24 polígonos")
    if len(new_train) != 64 or len(new_test) != 24:
        raise ValueError("La división actual debe tener 64/24 polígonos")

    return {
        "old": old,
        "new": new,
        "old_train": old_train,
        "old_test": old_test,
        "new_train": new_train,
        "new_test": new_test,
        "common": old_test - {"5_03", "5_11"},
    }


def fit_model(key, data, train_ids):
    train = data.loc[data["poly_id"].isin(train_ids)]
    if train.empty:
        raise ValueError(f"Sin píxeles de entrenamiento para {key}")
    X = train[BANDS].to_numpy(dtype=float) / 10000.0
    y = train["class_id"].to_numpy(dtype=int)
    model = pipeline(MODELS[key][1])
    model.fit(X, y)
    return model


def evaluate(model, data, eval_ids):
    test = data.loc[data["poly_id"].isin(eval_ids)]
    if test.empty:
        raise ValueError("Sin píxeles de evaluación")
    X = test[BANDS].to_numpy(dtype=float) / 10000.0
    y = test["class_id"].to_numpy(dtype=int)
    prediction = model.predict(X)
    return y, prediction


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


def per_class(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return pd.DataFrame({
        "class_id": LABELS,
        "class_name": NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def tex_number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", "{,}")


def tex_delta(value):
    if abs(value) < 0.0005:
        return "$0{,}000$"
    sign = "+" if value > 0 else "-"
    return f"${sign}{tex_number(abs(value))}$"


def validate_against_archive(old_metrics, new_metrics):
    archived = pd.read_csv(BASE_METRICS).set_index("model")
    current = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model")
    for key in MODELS:
        if key not in archived.index or key not in current.index:
            raise ValueError(f"Falta {key} en las métricas archivadas")
        for metric in ("oa", "kappa", "f1_macro"):
            if abs(old_metrics[key][metric] - archived.loc[key, metric]) > 1e-9:
                raise ValueError(f"La base de 72 no reproduce {key}/{metric}")
            if abs(new_metrics[key][metric] - current.loc[key, metric]) > 1e-9:
                raise ValueError(f"El dataset actual no reproduce {key}/{metric}")


def write_table(paired):
    lines = [
        r"\begin{table}[htbp]",
        r" \centering",
        r" \small",
        r" \setlength{\tabcolsep}{4pt}",
        (r" \caption{Comparación pareada del efecto de la ampliación (OA). "
         r"\emph{Original 24}: test de la versión de 72 polígonos. "
         r"\emph{Nuevo 24}: test de la versión de 88. "
         r"\emph{Común 22}: test original sin \texttt{5\_03} ni "
         r"\texttt{5\_11}. La base del test nuevo se entrena solo con "
         r"polígonos originales que no están en ese test, para evitar fuga.}"),
        r" \label{tab:comparacion_pareada}",
        r" \begin{tabular}{lcccc}",
        r"  \toprule",
        r"  Test & Modelo & Base & Nuevo & $\Delta$ \\",
        r"  \midrule",
    ]
    for row in paired.itertuples():
        lines.append(
            f"  {row.test} & {row.model} & {tex_number(row.base_oa)} & "
            f"{tex_number(row.new_oa)} & {tex_delta(row.delta_oa)} \\\\"
        )
    lines += [r"  \bottomrule", r" \end{tabular}", r"\end{table}", ""]
    text = "\n".join(lines)
    TAB2.mkdir(parents=True, exist_ok=True)
    TAB_FULL.mkdir(parents=True, exist_ok=True)
    (TAB2 / "comparacion_pareada.tex").write_text(text, encoding="utf-8")
    (TAB_FULL / "comparacion_pareada.tex").write_text(text, encoding="utf-8")


def main():
    data = load_data()
    cache = {}

    def get_model(key, version, train_ids):
        cache_key = (key, version, frozenset(train_ids))
        if cache_key not in cache:
            cache[cache_key] = fit_model(key, data[version], train_ids)
        return cache[cache_key]

    old_metrics = {}
    new_metrics = {}
    for key in MODELS:
        old_model = get_model(key, "old", data["old_train"])
        new_model = get_model(key, "new", data["new_train"])
        old_metrics[key] = metrics(
            *evaluate(old_model, data["old"], data["old_test"])
        )
        new_metrics[key] = metrics(
            *evaluate(new_model, data["new"], data["new_test"])
        )

    validate_against_archive(old_metrics, new_metrics)

    comparison = pd.DataFrame([
        {
            "model": key,
            "oa_antes": old_metrics[key]["oa"],
            "oa_despues": new_metrics[key]["oa"],
            "delta_oa": new_metrics[key]["oa"] - old_metrics[key]["oa"],
            "kappa_antes": old_metrics[key]["kappa"],
            "kappa_despues": new_metrics[key]["kappa"],
            "f1_antes": old_metrics[key]["f1_macro"],
            "f1_despues": new_metrics[key]["f1_macro"],
        }
        for key in MODELS
    ]).round(6)
    comparison.to_csv(RESULTS / "expansion_comparison.csv", index=False)

    per_class_rows = []
    for key in ("SVM", "ANN"):
        old_model = get_model(key, "old", data["old_train"])
        new_model = get_model(key, "new", data["new_train"])
        old_y, old_pred = evaluate(old_model, data["old"], data["old_test"])
        new_y, new_pred = evaluate(new_model, data["new"], data["new_test"])
        old_pc = per_class(old_y, old_pred).set_index("class_id")
        new_pc = per_class(new_y, new_pred).set_index("class_id")
        for class_id in (3, 4, 5, 6):
            per_class_rows.append({
                "model": key,
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "recall_antes": old_pc.loc[class_id, "recall"],
                "recall_despues": new_pc.loc[class_id, "recall"],
                "f1_antes": old_pc.loc[class_id, "f1"],
                "f1_despues": new_pc.loc[class_id, "f1"],
            })
    pd.DataFrame(per_class_rows).round(6).to_csv(
        RESULTS / "expansion_per_class.csv", index=False
    )

    scenarios = [
        ("Original 24", data["old_train"], data["old"], data["old_test"]),
        ("Nuevo 24", data["old_train"], data["new"], data["new_test"]),
        ("Común 22", data["old_train"], data["old"], data["common"]),
    ]
    paired_rows = []
    for label, base_train, eval_data, eval_ids in scenarios:
        for key in ("LR", "SVM"):
            base_model = get_model(key, "old", base_train - eval_ids)
            new_model = get_model(key, "new", data["new_train"] - eval_ids)
            base_y, base_pred = evaluate(base_model, eval_data, eval_ids)
            new_y, new_pred = evaluate(new_model, eval_data, eval_ids)
            base_m = metrics(base_y, base_pred)
            new_m = metrics(new_y, new_pred)
            paired_rows.append({
                "test": label,
                "model": key,
                "base_oa": base_m["oa"],
                "new_oa": new_m["oa"],
                "delta_oa": new_m["oa"] - base_m["oa"],
                "base_kappa": base_m["kappa"],
                "new_kappa": new_m["kappa"],
                "base_f1": base_m["f1_macro"],
                "new_f1": new_m["f1_macro"],
            })

    paired = pd.DataFrame(paired_rows).round(6)
    paired.to_csv(RESULTS / "expansion_paired.csv", index=False)
    write_table(paired)

    print("Comparación de la ampliación")
    print(comparison.to_string(index=False))
    print("\nComparación pareada (OA):")
    print(paired[["test", "model", "base_oa", "new_oa", "delta_oa"]].to_string(index=False))
    print("\nArchivos regenerados en", RESULTS, "y", TAB2)


if __name__ == "__main__":
    main()
