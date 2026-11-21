"""Fase 1 - Figuras y tablas LaTeX para el reporte.

Uso:  run.bat src\\02_figures.py
"""
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (BAND_INFO, BANDS, CLASS_COLORS, CLASS_LABELS as CLASSES, FIG, METADATA,
                    POLYGONS, ROI, SCL, SCL_INVALID, STACK,
                    STUDY_AREA, TAB, TARGET_EPSG, TSV)

plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "savefig.dpi": 150})


def stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, (lo, hi))
    return np.clip((x - a) / (b - a), 0, 1)


def km_axes(ax, dec=0):
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.{dec}f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.{dec}f}")
    ax.set_xlabel("Este (km, UTM 23S)")
    ax.set_ylabel("Norte (km, UTM 23S)")


def scalebar(ax, ext, km=2):
    x0 = ext[0] + 0.03 * (ext[1] - ext[0]); y0 = ext[2] + 0.06 * (ext[3] - ext[2])
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="white", lw=3)
    ax.text(x0 + km * 500, y0 + 120, f"{km} km", color="white", ha="center", fontsize=8)
    ax.annotate("N", xy=(ext[1] - 400, ext[3] - 300), xytext=(ext[1] - 400, ext[3] - 1300),
                color="white", ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="white"))


def tex_escape(s):
    return str(s).replace("_", r"\_\allowbreak{}").replace("%", r"\%").replace("&", r"\&")


def dec(x, n=2):
    """Número con coma decimal (estilo del reporte en español)."""
    return f"{x:.{n}f}".replace(".", "{,}")


def write_table(name, caption, label, env, spec, header, rows, pre=""):
    """Escribe un float table completo en report/tables/<name>.tex."""
    import re
    width = "{\\textwidth}" if env == "tabularx" else ""
    header = [re.sub(r"^\\textbf\{(.*)\}$", r"\1", h) for h in header]  # sin negrita
    lines = ["\\begin{table}[htbp]", " \\centering"]
    if pre:
        lines.append(" " + pre)
    lines += [f" \\caption{{{caption}.}}", f" \\label{{{label}}}",
              f" \\begin{{{env}}}{width}{{{spec}}}", "  \\toprule",
              "  " + " & ".join(header) + " \\\\", "  \\midrule"]
    lines += ["  " + r for r in rows]
    lines += ["  \\bottomrule", f" \\end{{{env}}}", "\\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    with rasterio.open(STACK) as ds:
        transform, w, h = ds.transform, ds.width, ds.height
        a = ds.read().astype("float32") / 10000
    B = {b: a[i] for i, b in enumerate(BANDS)}
    ext = [transform.c, transform.c + transform.a * w,
           transform.f + transform.e * h, transform.f]
    tc = np.dstack([stretch(B["B04"]), stretch(B["B03"]), stretch(B["B02"])])
    fc = np.dstack([stretch(B["B08"]), stretch(B["B04"]), stretch(B["B03"])])
    swir = np.dstack([stretch(B["B12"]), stretch(B["B08"]), stretch(B["B04"])])
    roi = gpd.read_file(ROI).to_crs(TARGET_EPSG)
    sa = gpd.read_file(STUDY_AREA).to_crs(TARGET_EPSG)
    polys_src = POLYGONS
    polys = gpd.read_file(polys_src).to_crs(TARGET_EPSG)

    # 1. Área de estudio en color verdadero
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(tc, extent=ext)
    sa.boundary.plot(ax=ax, color="white", lw=1.2, ls="--")
    roi.boundary.plot(ax=ax, color="yellow", lw=1)
    for x, y, t in [(582900, 7772300, "Brumadinho"), (591200, 7775700, "Mina Córrego\ndo Feijão"),
                    (585500, 7771300, "Río Paraopeba")]:
        ax.text(x, y, t, color="white", fontsize=8, ha="center",
                bbox=dict(fc="black", alpha=0.45, lw=0, pad=1.5))
    scalebar(ax, ext); km_axes(ax)
    ax.legend(handles=[Line2D([], [], color="white", ls="--", label="Área de estudio"),
                       Line2D([], [], color="yellow", label="ROI corredor de relaves (midterm)")],
              loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title("Sentinel-2 L2A, 01-02-2019, color verdadero (B04-B03-B02)")
    fig.savefig(FIG / "area_estudio_tc.jpg", pil_kwargs={"quality": 85}); plt.close(fig)

    # 2. Composiciones en falso color
    fig, axs = plt.subplots(2, 1, figsize=(10, 9))
    for ax, img, t in [(axs[0], fc, "Falso color infrarrojo (B08-B04-B03)"),
                       (axs[1], swir, "Falso color SWIR (B12-B08-B04)")]:
        ax.imshow(img, extent=ext); roi.boundary.plot(ax=ax, color="yellow", lw=0.8)
        ax.set_title(t); km_axes(ax); scalebar(ax, ext)
    fig.savefig(FIG / "falso_color.jpg", pil_kwargs={"quality": 85}); plt.close(fig)

    # 3. Máscara SCL
    with rasterio.open(SCL) as ds:
        scl = ds.read(1)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(tc * 0.55, extent=ext)
    inv = np.ma.masked_where(~np.isin(scl, SCL_INVALID), np.ones_like(scl))
    ax.imshow(inv, extent=ext, cmap=matplotlib.colors.ListedColormap(["red"]), alpha=0.9)
    ax.set_title(f"Píxeles inválidos según SCL (clases 0,1,3,8,9,10): "
                 f"{meta['aoi_invalid_scl_pct']:.2f} % del área")
    km_axes(ax); fig.savefig(FIG / "scl_mask.jpg", pil_kwargs={"quality": 85}); plt.close(fig)

    # 4. Polígonos de entrenamiento
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(tc, extent=ext)
    for cid, g in polys.groupby("class_id"):
        g.plot(ax=ax, facecolor=CLASS_COLORS[cid], edgecolor="white", lw=0.4)
        c = g.geometry.centroid
        ax.scatter(c.x, c.y, s=22, color=CLASS_COLORS[cid], edgecolor="white", lw=0.7, zorder=3)
    roi.boundary.plot(ax=ax, color="yellow", lw=0.6)
    ax.legend(handles=[Patch(fc=CLASS_COLORS[k], ec="white", label=f"{k}. {v}")
                       for k, v in CLASSES.items()], loc="upper left", fontsize=7, ncol=2,
              framealpha=0.85)
    ax.set_title(f"Polígonos de entrenamiento ({len(polys)} parcelas)")
    km_axes(ax); scalebar(ax, ext)
    fig.savefig(FIG / "poligonos_entrenamiento.jpg", pil_kwargs={"quality": 85}); plt.close(fig)

    # 4b. Zoom: corredor del relave y casco urbano
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (x0, x1, y0, y1), t in [(axs[0], (588000, 592800, 7771000, 7775800), "Mina y corredor de relaves"),
                                    (axs[1], (581040, 584800, 7770600, 7774300), "Brumadinho y río Paraopeba")]:
        ax.imshow(fc, extent=ext)
        for cid, g in polys.groupby("class_id"):
            g.plot(ax=ax, facecolor="none", edgecolor=CLASS_COLORS[cid], lw=1.6)
        roi.boundary.plot(ax=ax, color="yellow", lw=0.6, ls=":")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_title(t); km_axes(ax, 1)
    axs[1].legend(handles=[Patch(fc="none", ec=CLASS_COLORS[k], lw=1.6, label=v)
                           for k, v in CLASSES.items()], fontsize=7, loc="upper right")
    fig.savefig(FIG / "poligonos_zoom.jpg", pil_kwargs={"quality": 85}); plt.close(fig)

    # 5. Firmas espectrales medias por clase
    df = pd.read_csv(TSV, sep="\t")
    wl = np.arange(len(BANDS))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cid, g in df.groupby("class_id"):
        m = g[BANDS].mean() / 10000; s = g[BANDS].std() / 10000
        ax.plot(wl, m, "-o", ms=3, color=CLASS_COLORS[cid], label=CLASSES[cid])
        ax.fill_between(wl, m - s, m + s, color=CLASS_COLORS[cid], alpha=0.15)
    ax.set_xticks(wl)
    ax.set_xticklabels([f"{b}\n{BAND_INFO[b][1]}" for b in BANDS], fontsize=7)
    ax.set_xlabel("Banda y longitud de onda central (nm)")
    ax.set_ylabel("Reflectancia de superficie")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Firma espectral media (±1σ) por clase")
    fig.savefig(FIG / "firmas_espectrales.png"); plt.close(fig)

    # ---- Tablas LaTeX ----
    rows = [
        ("Identificador del producto", "\\texttt{" + tex_escape(meta["product_uri"]) + "}"),
        ("Plataforma / sensor", f"{meta['platform']} / MSI"),
        ("Nivel de procesamiento", "L2A (Sen2Cor), reflectancia BOA"),
        ("\\emph{Processing baseline}", meta["processing_baseline"]),
        ("Fecha y hora de adquisición (UTC)", meta["datetime_utc"].replace("T", " ")[:19]),
        ("Tile MGRS / órbita relativa", f"{meta['mgrs_tile']} / R{meta['relative_orbit']:03d} (descendente)"),
        ("Nubosidad de la escena completa", f"{dec(meta['scene_cloud_cover_pct'])}\\,\\%"),
        ("Píxeles inválidos (SCL) en el área", f"{dec(meta['aoi_invalid_scl_pct'])}\\,\\%"),
        ("Ángulos solares medios", f"cenital {dec(meta['sun_zenith'], 1)}$^\\circ$, azimut {dec(meta['sun_azimuth'], 1)}$^\\circ$"),
        ("SRC del stack", "WGS 84 / UTM zona 23S (EPSG:32723)"),
        ("Tamaño de píxel", "10\\,m $\\times$ 10\\,m"),
        ("Dimensiones", f"{meta['width']} columnas $\\times$ {meta['height']} filas"),
        ("Tipo de dato / escala", "UInt16, reflectancia $\\times$ 10\\,000 (sin offset)"),
        ("Fuente", "Microsoft Planetary Computer (STAC, \\texttt{sentinel-2-l2a})"),
    ]
    # Cada archivo contiene la tabla COMPLETA (float + tabular + reglas).
    # No se usa \input dentro de un tabular: desde LaTeX 2020 un \input seguido
    # de \bottomrule produce "Misplaced \noalign".
    write_table("metadatos", "Metadatos de la escena utilizada", "tab:meta",
                "tabularx", "lX", ["\\textbf{Campo}", "\\textbf{Valor}"],
                [f"{k} & {v} \\\\" for k, v in rows], pre="\\small")

    write_table("bandas", "Bandas Sentinel-2 L2A utilizadas \\parencite{Drusch2012}",
                "tab:bandas", "tabular", "llrrl",
                ["\\textbf{Banda}", "\\textbf{Región}", "\\textbf{$\\lambda$ central (nm)}",
                 "\\textbf{Res. nativa (m)}", "\\textbf{Remuestreo}"],
                [f"{b} & {BAND_INFO[b][0]} & {BAND_INFO[b][1]} & {BAND_INFO[b][2]} & "
                 f"{'---' if BAND_INFO[b][2] == 10 else 'Bilineal a 10 m'} \\\\" for b in BANDS])

    cnt = df.groupby("class_id").agg(p=("poly_id", "nunique"), n=("poly_id", "size"))
    tot_n = int(cnt.n.sum()); tot_p = int(cnt.p.sum())
    lines = [f"{cid} & {CLASSES[cid]} & {int(cnt.loc[cid, 'p'])} & {int(cnt.loc[cid, 'n'])} & "
             f"{dec(100 * cnt.loc[cid, 'n'] / tot_n, 1)}\\,\\% \\\\" for cid in cnt.index]
    lines += ["\\midrule", f" & \\textbf{{Total}} & {tot_p} & {tot_n} & 100\\,\\% \\\\"]
    write_table("conteo_clases",
                "Polígonos y píxeles por clase en \\texttt{training\\_dataset.tsv}",
                "tab:conteo", "tabular", "clrrr",
                ["\\textbf{Código}", "\\textbf{Clase}", "\\textbf{Polígonos}",
                 "\\textbf{Píxeles}", "\\textbf{\\%}"], lines)

    head = df.groupby("class_id").head(1).head(6)
    cols = ["Latitude", "Longitude"] + BANDS + ["class_name"]
    write_table("tsv_muestra", "Primeras filas del TSV (una por clase)", "tab:muestra",
                "tabular", "r" * 14 + "l",
                [f"\\textbf{{{c}}}" for c in ["Lat", "Lon"] + BANDS + ["Clase"]],
                [" & ".join([dec(r.Latitude, 5), dec(r.Longitude, 5)]
                            + [str(int(r[b])) for b in BANDS] + [r.class_name]) + " \\\\"
                 for _, r in head[cols].iterrows()],
                pre="\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}")

    summary = {"filas": tot_n, "poligonos": tot_p, "poligonos_fuente": polys_src.name}
    (TAB / "resumen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Figuras en", FIG); print("Tablas en", TAB); print(summary)


if __name__ == "__main__":
    main()
