"""Fase 3: figuras y tablas del reporte.

Genera las figuras de HPO, comparación con Fase 2, matriz de confusión,
importancia por permutación, mapa temático, zoom del relave y áreas.

Uso:
    run.bat src\\07_phase3_figures.py
"""

import json

import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from common import (BANDS, CLASS_COLORS, CLASS_LABELS, CLASSES, FIG3, FINAL_MODEL,
                    MODEL_CARD, PREDICTION, RESULTS, ROI, STUDY_AREA, TAB3, TSV)

SEED = 42
MODEL_ORDER = ["LR", "SVM", "ANN", "KNN", "RF", "DT"]


def dec(value, digits=3):
    return f"{value:.{digits}f}".replace(".", "{,}")


def write_table(name, caption, label, header, rows, align=None):
    align = align or ("l" + "c" * (len(header) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r" \centering",
        r" \small",
        r" \setlength{\tabcolsep}{4pt}",
        f" \\caption{{{caption}}}",
        f" \\label{{{label}}}",
        f" \\begin{{tabular}}{{{align}}}",
        r"  \toprule",
        "  " + " & ".join(header) + r" \\",
        r"  \midrule",
    ]
    lines += ["  " + " & ".join(row) + r" \\" for row in rows]
    lines += [r"  \bottomrule", r" \end{tabular}", r"\end{table}", ""]
    (TAB3 / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def figure_hpo():
    cv = pd.read_csv(RESULTS / "phase3_cv_summary.csv").set_index("model")
    order = [model for model in MODEL_ORDER if model in cv.index]
    values = cv.loc[order, "best_kappa"]
    errors = cv.loc[order, "best_kappa_std"]
    colors = ["#2166ac" if model == "LR" else "#9ecae1" for model in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(order, values, yerr=errors, capsize=4, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Kappa medio en CV agrupada")
    ax.set_title("Optimización de hiperparámetros por familia")
    ax.grid(axis="y", alpha=0.3)
    for index, value in enumerate(values):
        ax.text(index, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG3 / "hpo_modelos.png", dpi=300)
    plt.close(fig)


def figure_tuned_vs_initial():
    initial = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model")
    tuned = pd.read_csv(RESULTS / "phase3_test_metrics.csv").set_index("model")
    order = [model for model in MODEL_ORDER if model in initial.index and model in tuned.index]
    x = np.arange(len(order))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    for ax, metric, title in zip(
        axes, ("oa", "kappa"), ("Exactitud global", "Kappa")
    ):
        ax.bar(x - width / 2, initial.loc[order, metric], width,
               label="Fase 2", color="#bdbdbd", edgecolor="black", linewidth=0.4)
        ax.bar(x + width / 2, tuned.loc[order, metric], width,
               label="Fase 3", color="#3182bd", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylim(0.7, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Valor")
    axes[0].legend(frameon=False)
    fig.suptitle("Modelos iniciales frente a modelos optimizados")
    fig.tight_layout()
    fig.savefig(FIG3 / "tuned_vs_inicial.png", dpi=300)
    plt.close(fig)


def figure_confusion():
    matrix = pd.read_csv(RESULTS / "phase3_confusion_LR.csv", index_col=0)
    row_labels = [CLASSES[class_id] for class_id in sorted(CLASSES)]
    labels = [CLASS_LABELS[class_id] for class_id in sorted(CLASSES)]
    matrix = matrix.loc[row_labels, row_labels]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Referencia")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix.iloc[row, col])
            ax.text(col, row, value, ha="center", va="center",
                    color="white" if value > matrix.to_numpy().max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Píxeles")
    ax.set_title("Matriz de confusión del modelo final")
    fig.tight_layout()
    fig.savefig(FIG3 / "confusion_final.png", dpi=300)
    plt.close(fig)


def figure_importance():
    artifact = joblib.load(FINAL_MODEL)
    data = pd.read_csv(TSV, sep="\t")
    split = pd.read_csv(RESULTS / "split.csv")
    train_ids = set(split.loc[split["set"] == "entrenamiento", "poly_id"])
    test_ids = set(split.loc[split["set"] == "prueba", "poly_id"])
    train = data.loc[data["poly_id"].isin(train_ids)]
    test = data.loc[data["poly_id"].isin(test_ids)]
    model = clone(artifact["model"])
    model.fit(train[BANDS].to_numpy(dtype=float) / 10000.0,
              train["class_id"].to_numpy(dtype=int))
    result = permutation_importance(
        model,
        test[BANDS].to_numpy(dtype=float) / 10000.0,
        test["class_id"].to_numpy(dtype=int),
        n_repeats=30,
        random_state=SEED,
        scoring="accuracy",
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": BANDS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=True)
    importance.to_csv(RESULTS / "phase3_permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(importance["feature"], importance["importance_mean"],
            xerr=importance["importance_std"], color="#3182bd",
            edgecolor="black", linewidth=0.4, capsize=3)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Cambio medio en exactitud al permutar")
    ax.set_title("Importancia por permutación del modelo final")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG3 / "importancia_final.png", dpi=300)
    plt.close(fig)


def map_layers(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        extent = (
            src.transform.c,
            src.transform.c + src.width * src.transform.a,
            src.transform.f + src.height * src.transform.e,
            src.transform.f,
        )
        crs = src.crs
    colors = ["#f0f0f0"] + [CLASS_COLORS[class_id] for class_id in sorted(CLASSES)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), len(colors))
    study = gpd.read_file(STUDY_AREA).to_crs(crs)
    roi = gpd.read_file(ROI).to_crs(crs)
    return data, extent, cmap, norm, study, roi


def figure_map(tif_path):
    data, extent, cmap, norm, study, roi = map_layers(tif_path)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(data, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    study.boundary.plot(ax=ax, color="black", linewidth=0.8)
    roi.boundary.plot(ax=ax, color="#f1c40f", linewidth=0.8, linestyle="--")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Patch(facecolor="#f0f0f0", edgecolor="black", label="Sin datos")]
    handles += [
        Patch(facecolor=CLASS_COLORS[class_id], edgecolor="black",
              label=CLASS_LABELS[class_id])
        for class_id in sorted(CLASSES)
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=7)
    ax.set_title("Mapa temático de coberturas")
    fig.tight_layout()
    fig.savefig(FIG3 / "mapa_tematico.png", dpi=300)
    plt.close(fig)


def figure_zoom(tif_path):
    data, extent, cmap, norm, study, roi = map_layers(tif_path)
    xmin, ymin, xmax, ymax = roi.total_bounds
    pad_x = 0.08 * (xmax - xmin)
    pad_y = 0.08 * (ymax - ymin)
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.imshow(data, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    study.boundary.plot(ax=ax, color="black", linewidth=0.8)
    roi.boundary.plot(ax=ax, color="#f1c40f", linewidth=0.9, linestyle="--")
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Detalle del corredor del relave")
    fig.tight_layout()
    fig.savefig(FIG3 / "mapa_zoom_relave.png", dpi=300)
    plt.close(fig)


def figure_area():
    area = pd.read_csv(RESULTS / "phase3_area_by_class.csv")
    labels = [CLASS_LABELS[int(class_id)] for class_id in area["class_id"]]
    colors = [CLASS_COLORS[int(class_id)] for class_id in area["class_id"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(labels, area["km2"], color=colors, edgecolor="black", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Área (km²)")
    ax.set_title("Área predicha por clase")
    ax.grid(axis="x", alpha=0.3)
    for bar, value, percent in zip(bars, area["km2"], area["percent_valid"]):
        ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} km² ({percent:.1f} %)", va="center", fontsize=8)
    ax.set_xlim(0, area["km2"].max() * 1.3)
    fig.tight_layout()
    fig.savefig(FIG3 / "area_clases.png", dpi=300)
    plt.close(fig)


def tables():
    cv = pd.read_csv(RESULTS / "phase3_cv_summary.csv").sort_values(
        "best_kappa", ascending=False
    )
    rows = [
        [row.model, dec(row.best_kappa), dec(row.best_f1_macro),
         str(int(row.n_candidates)), "Sí" if row.model == "LR" else "No"]
        for row in cv.itertuples()
    ]
    write_table(
        "hpo_resumen",
        "Mejor Kappa de validación agrupada por familia. La columna final marca "
        "la familia seleccionada.",
        "tab:hpo",
        ["Modelo", "Kappa CV", "F1 CV", "Candidatos", "Seleccionado"],
        rows,
    )

    metrics = pd.read_csv(RESULTS / "phase3_test_metrics.csv").sort_values(
        "kappa", ascending=False
    )
    rows = [
        [row.model, dec(row.oa), dec(row.kappa), dec(row.f1_macro),
         dec(row.cv_kappa), dec(row.cv_family_best_kappa),
         "Sí" if row.selected_by_cv else "No"]
        for row in metrics.itertuples()
    ]
    write_table(
        "phase3_metricas",
        "Métricas en el test congelado de 24 polígonos y Kappa de validación "
        "agrupada. \\emph{CV familia} es el mejor Kappa de la familia; \\emph{CV "
        "modelo} es el del conjunto de hiperparámetros elegido.",
        "tab:phase3_metricas",
        ["Modelo", "OA", "Kappa", "F1", "CV modelo", "CV familia", "Final"],
        rows,
    )

    card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
    rows = []
    for key, value in card["params"].items():
        rows.append([key.replace("clf__", "").replace("_", r"\_"), str(value)])
    write_table(
        "params_finales",
        f"Hiperparámetros del modelo final ({card['selected_name']}).",
        "tab:params_finales",
        ["Parámetro", "Valor"],
        rows,
        align="ll",
    )

    area = pd.read_csv(RESULTS / "phase3_area_by_class.csv")
    rows = [
        [CLASS_LABELS[int(row.class_id)], f"{int(row.pixels):,}".replace(",", "."),
         dec(row.hectares, 1), dec(row.km2), dec(row.percent_valid, 1) + r"\,\%"]
        for row in area.itertuples()
    ]
    write_table(
        "area_clases",
        "Área predicha por clase. Los porcentajes se calculan sobre el área "
        "válida, no sobre el rectángulo completo.",
        "tab:area_clases",
        ["Clase", "Píxeles", "Hectáreas", "km²", "\\% válido"],
        rows,
    )

    corridor = pd.read_csv(RESULTS / "phase3_area_by_corridor.csv")
    rows = [
        [CLASS_LABELS[int(row.class_id)], f"{int(row.pixels):,}".replace(",", "."),
         dec(row.km2)]
        for row in corridor.itertuples()
    ]
    write_table(
        "area_corredor",
        "Área predicha dentro del corredor del relave. El corredor tiene "
        "5{,}38\\,km$^2$; el resto de la clase relave queda fuera de sus límites.",
        "tab:area_corredor",
        ["Clase", "Píxeles", "km²"],
        rows,
    )


def main():
    FIG3.mkdir(parents=True, exist_ok=True)
    TAB3.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(PREDICTION.glob("landcover_*.tif"))
    if not tif_files:
        raise FileNotFoundError("No hay GeoTIFF en data/prediction")
    tif_path = tif_files[0]

    figure_hpo()
    figure_tuned_vs_initial()
    figure_confusion()
    figure_importance()
    figure_map(tif_path)
    figure_zoom(tif_path)
    figure_area()
    tables()
    print("Figuras en", FIG3)
    print("Tablas en", TAB3)
    print("GeoTIFF usado:", tif_path)


if __name__ == "__main__":
    main()
