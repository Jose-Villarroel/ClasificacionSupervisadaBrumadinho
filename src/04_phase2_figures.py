"""Fase 2 - Figuras y tablas LaTeX a partir de results/ (salida de 03_train_models.py).

Usa el Python de QGIS porque el mapa de la división necesita GDAL y geopandas.
Uso:  run.bat src\\04_phase2_figures.py
"""
import importlib
import json
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch

from common import (BAND_INFO, BANDS, CLASS_COLORS, CLASS_LABELS, CLASSES, FIG2,
                    POLYGONS, RESULTS, ROI, STACK, TAB2, TARGET_EPSG, TSV)

f1 = importlib.import_module("02_figures")       # reutiliza estilo y utilidades de la Fase 1
stretch, km_axes, scalebar, dec = f1.stretch, f1.km_axes, f1.scalebar, f1.dec

ORDER = ["LR", "DT", "RF", "SVM", "ANN", "KNN"]
SHORT = {1: "Veg. densa", 2: "Veg. baja", 3: "Agua", 4: "Relave", 5: "Suelo exp.", 6: "Urbano"}


def write_table(name, caption, label, env, spec, header, rows, pre=""):
    """Un float table completo por archivo (no usar \\input dentro de un tabular)."""
    width = "{\\textwidth}" if env == "tabularx" else ""
    lines = ["\\begin{table}[htbp]", " \\centering"] + ([" " + pre] if pre else [])
    lines += [f" \\caption{{{caption}.}}", f" \\label{{{label}}}",
              f" \\begin{{{env}}}{width}{{{spec}}}", "  \\toprule",
              "  " + " & ".join(header) + " \\\\", "  \\midrule"]
    lines += ["  " + r for r in rows]
    lines += ["  \\bottomrule", f" \\end{{{env}}}", "\\end{table}", ""]
    (TAB2 / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def load():
    summ = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model").loc[ORDER]
    cms = {m: pd.read_csv(RESULTS / f"confusion_{m}.csv", index_col=0) for m in ORDER}
    per = {m: pd.read_csv(RESULTS / f"per_class_{m}.csv") for m in ORDER}
    split = pd.read_csv(RESULTS / "split.csv")
    imp = pd.read_csv(RESULTS / "feature_importance.csv", index_col=0)
    return summ, cms, per, split, imp


def best_model(summ):
    return summ.sort_values(["kappa", "f1_macro"], ascending=False).index[0]


def fig_confusion(summ, cms):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.2))
    for ax, m in zip(axs.flat, ORDER):
        cm = cms[m].to_numpy()
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.55 else "black")
        lab = [SHORT[k] for k in range(1, 7)]
        ax.set_xticks(range(6)); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6)); ax.set_yticklabels(lab, fontsize=7)
        ax.set_title(f"{m}: OA = {summ.loc[m, 'oa'] * 100:.1f} %, κ = {summ.loc[m, 'kappa']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Clase predicha", fontsize=8); ax.set_ylabel("Clase real", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG2 / "matrices_confusion.png"); plt.close(fig)


def fig_metrics(summ):
    x = np.arange(len(ORDER)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2, summ["oa"], w, label="OA (prueba)", color="#2166ac")
    ax.bar(x + w / 2, summ["kappa"], w, label="Kappa (prueba)", color="#92c5de")
    ax.errorbar(x - w / 2, summ["cv_oa_mean"], yerr=summ["cv_oa_std"], fmt="o", color="black",
                ms=4, capsize=3, label="OA validación cruzada (media ± σ)")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    lo = max(0, min(summ["kappa"].min(), (summ["cv_oa_mean"] - summ["cv_oa_std"]).min()) - 0.05)
    ax.set_ylim(lo, 1.0); ax.set_ylabel("Valor"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax.set_title("Exactitud global y Kappa por modelo")
    fig.savefig(FIG2 / "metricas_modelos.png"); plt.close(fig)


def fig_importance(imp):
    x = np.arange(len(BANDS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, imp["DT"], w, label="Árbol de decisión", color="#8c510a")
    ax.bar(x + w / 2, imp["RF"], w, label="Random Forest", color="#1b7837")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{BAND_INFO[b][1]}" for b in BANDS], fontsize=7)
    ax.set_xlabel("Banda y longitud de onda central (nm)")
    ax.set_ylabel("Importancia (reducción de impureza)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Importancia de las bandas")
    fig.savefig(FIG2 / "importancia_bandas.png"); plt.close(fig)


def fig_split_map(split):
    with rasterio.open(STACK) as ds:
        transform, w, h = ds.transform, ds.width, ds.height
        rgb = np.dstack([stretch(ds.read(i).astype("float32")) for i in (4, 3, 2)])
    ext = [transform.c, transform.c + transform.a * w,
           transform.f + transform.e * h, transform.f]
    polys = gpd.read_file(POLYGONS).to_crs(TARGET_EPSG).merge(split[["poly_id", "set"]], on="poly_id")
    roi = gpd.read_file(ROI).to_crs(TARGET_EPSG)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(rgb * 0.8, extent=ext)
    roi.boundary.plot(ax=ax, color="yellow", lw=0.6)
    for (cid, s), g in polys.groupby(["class_id", "set"]):
        c = g.geometry.centroid
        ax.scatter(c.x, c.y, s=40 if s == "prueba" else 26, color=CLASS_COLORS[cid],
                   marker="s" if s == "prueba" else "o",
                   edgecolor="black" if s == "prueba" else "white", lw=1.0, zorder=3)
    hs = [Patch(fc=CLASS_COLORS[k], ec="white", label=v) for k, v in CLASS_LABELS.items()]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="grey", mec="white", label="Entrenamiento (8 por clase)"),
           plt.Line2D([], [], marker="s", ls="", mfc="grey", mec="black", label="Prueba (4 por clase)")]
    ax.legend(handles=hs, loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title("División de los polígonos en entrenamiento y prueba")
    km_axes(ax); scalebar(ax, ext)
    fig.savefig(FIG2 / "division_poligonos.jpg", pil_kwargs={"quality": 85}); plt.close(fig)


def tables(summ, cms, per, split, best):
    # 1. Resumen de métricas, mejor valor en negrita
    cols = ["oa", "kappa", "precision_macro", "recall_macro", "f1_macro"]
    best_val = {c: summ[c].max() for c in cols}

    def cell(m, c):
        v = dec(summ.loc[m, c], 3)
        return f"\\textbf{{{v}}}" if np.isclose(summ.loc[m, c], best_val[c]) else v

    rows = [f"{m} & {summ.loc[m, 'name']} & " + " & ".join(cell(m, c) for c in cols)
            + f" & {dec(summ.loc[m, 'cv_oa_mean'], 3)} $\\pm$ {dec(summ.loc[m, 'cv_oa_std'], 3)} \\\\"
            for m in ORDER]
    write_table("metricas_resumen", "Métricas iniciales sobre el conjunto de prueba (24 polígonos) y "
                "exactitud global de la validación cruzada agrupada por polígono. En negrita, el mejor valor",
                "tab:metricas", "tabular", "llcccccc",
                ["", "Modelo", "OA", "Kappa", "Precisión", "Recall", "F1", "OA VC"], rows,
                pre="\\small\n\\setlength{\\tabcolsep}{4pt}")

    # 2. Matriz de confusión del mejor modelo (conteos), con totales y exactitudes
    cm = cms[best].to_numpy()
    rows = []
    for i in range(6):
        pa = cm[i, i] / cm[i].sum()
        rows.append(f"{SHORT[i + 1]} & " + " & ".join(str(v) for v in cm[i])
                    + f" & {cm[i].sum()} & {dec(pa * 100, 1)} \\\\")
    ua = [cm[j, j] / cm[:, j].sum() if cm[:, j].sum() else 0 for j in range(6)]
    rows += ["\\midrule", "Total & " + " & ".join(str(v) for v in cm.sum(0)) + f" & {cm.sum()} & \\\\",
             "EU (\\%) & " + " & ".join(dec(u * 100, 1) for u in ua) + " & & \\\\"]
    write_table("matriz_mejor", f"Matriz de confusión del mejor modelo inicial ({summ.loc[best, 'name']}). "
                "Filas: clase real; columnas: clase predicha; EP: exactitud del productor; EU: exactitud del usuario",
                "tab:matriz_mejor", "tabular", "l" + "r" * 8,
                ["Real / Predicha"] + [SHORT[k] for k in range(1, 7)] + ["Total", "EP (\\%)"],
                rows, pre="\\small\n\\setlength{\\tabcolsep}{3.5pt}")

    # 3. Métricas por clase del mejor modelo
    p = per[best]
    rows = [f"{CLASS_LABELS[r.class_id]} & {dec(r.precision, 3)} & {dec(r.recall, 3)} & "
            f"{dec(r.f1, 3)} & {int(r.support)} \\\\" for r in p.itertuples()]
    write_table("por_clase_mejor", f"Precisión, recall y F1 por clase del mejor modelo inicial "
                f"({summ.loc[best, 'name']})", "tab:por_clase", "tabular", "lcccr",
                ["Clase", "Precisión", "Recall", "F1", "Píxeles de prueba"], rows)

    # 4. División entrenamiento / prueba
    g = split.groupby(["class_id", "set"]).agg(p=("poly_id", "size"), n=("n_pixels", "sum")).unstack("set")
    rows = [f"{CLASS_LABELS[c]} & {g.loc[c, ('p', 'entrenamiento')]} & {g.loc[c, ('n', 'entrenamiento')]} & "
            f"{g.loc[c, ('p', 'prueba')]} & {g.loc[c, ('n', 'prueba')]} \\\\" for c in g.index]
    tot = g.sum()
    rows += ["\\midrule", f"Total & {tot[('p', 'entrenamiento')]} & {tot[('n', 'entrenamiento')]} & "
             f"{tot[('p', 'prueba')]} & {tot[('n', 'prueba')]} \\\\"]
    write_table("division", "División del conjunto de datos por polígono", "tab:division",
                "tabular", "lrrrr",
                ["Clase", "Polígonos entren.", "Píxeles entren.", "Polígonos prueba", "Píxeles prueba"], rows)

    # 5. Estadística del TSV: media ± desviación por banda y clase (DN)
    df = pd.read_csv(TSV, sep="\t")
    mean = df.groupby("class_id")[BANDS].mean(); std = df.groupby("class_id")[BANDS].std()
    rows = [f"{b} & " + " & ".join(f"{mean.loc[c, b]:.0f} $\\pm$ {std.loc[c, b]:.0f}" for c in range(1, 7))
            + " \\\\" for b in BANDS]
    write_table("estadistica_tsv", "Media $\\pm$ desviación estándar de cada banda por clase en "
                "\\texttt{training\\_dataset.tsv} (reflectancia $\\times$ 10\\,000)",
                "tab:estadistica", "tabular", "l" + "r" * 6,
                ["Banda"] + [SHORT[k] for k in range(1, 7)], rows,
                pre="\\footnotesize\n\\setlength{\\tabcolsep}{4pt}")

    # 6. Configuración inicial de los modelos
    cfg = {
        "LR": "\\texttt{max\\_iter=1000}, \\texttt{class\\_weight=balanced}",
        "DT": "Sin límite de profundidad, criterio Gini, \\texttt{class\\_weight=balanced}",
        "RF": "300 árboles, \\texttt{class\\_weight=balanced}",
        "SVM": "Kernel RBF, $C=1$, $\\gamma$ = \\texttt{scale}, \\texttt{class\\_weight=balanced}",
        "ANN": "Dos capas ocultas (64 y 32 neuronas), ReLU, Adam, \\texttt{max\\_iter=2000}",
        "KNN": "$k=5$, distancia euclidiana, voto uniforme",
    }
    rows = [f"{m} & {summ.loc[m, 'name']} & {cfg[m]} \\\\" for m in ORDER]
    write_table("modelos", "Modelos y configuración inicial (scikit-learn). En todos, las bandas se "
                "estandarizan con \\texttt{StandardScaler} ajustado solo con el entrenamiento",
                "tab:modelos", "tabularx", "llX", ["", "Modelo", "Configuración inicial"], rows,
                pre="\\small")


def main():
    FIG2.mkdir(parents=True, exist_ok=True); TAB2.mkdir(parents=True, exist_ok=True)
    summ, cms, per, split, imp = load()
    best = best_model(summ)
    fig_confusion(summ, cms); fig_metrics(summ); fig_importance(imp); fig_split_map(split)
    tables(summ, cms, per, split, best)
    q = json.loads((RESULTS / "tsv_quality.json").read_text(encoding="utf-8"))
    (TAB2 / "resumen.json").write_text(json.dumps({"mejor_modelo": best, **q}, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")
    print("Mejor modelo inicial:", best, "| figuras en", FIG2, "| tablas en", TAB2)


if __name__ == "__main__":
    main()
