"""Fase 2 - Dataset auxiliar por polígono.

Agrega el TSV de píxeles a una fila por polígono: mediana de cada banda,
rango intercuartílico, número de píxeles e índices espectrales. También hace
control de calidad del muestreo y de la geometría de los polígonos.

El entregable de la Fase 1 sigue siendo data/training_dataset.tsv (una fila
por píxel). Este archivo es auxiliar para el experimento por polígono.

Entradas: data/training_dataset.tsv, data/training_polygons.gpkg y
          results/polygon_sampling.json.
Salidas:  data/training_polygons.tsv y results/polygon_qc.json.

Uso:  run.bat src\\01b_polygon_dataset.py
"""
import json
from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd

from common import (BANDS, INDICES, POLYGONS, POLY_TSV, RESULTS, TARGET_EPSG,
                    TSV, add_indices)

MIN_PIXELS = 5
MAX_DROPPED_FRAC = 0.20
MAX_OUTLIER_FRAC = 0.30


def outlier_fraction(values):
    """Fracción de valores fuera de mediana ± 3·MAD (escala mínima de 1 DN)."""
    med = values.median()
    mad = (values - med).abs().median()
    scale = max(1.4826 * mad, 1.0)
    return float(((values - med).abs() > 3 * scale).mean())


def polygon_geometry_qc(polys):
    """Control de calidad de las geometrías, en EPSG:32723."""
    geoms = list(polys.geometry)
    invalid = int(sum(not g.is_valid for g in geoms))

    overlaps = []
    for (i, a), (j, b) in combinations(enumerate(geoms), 2):
        if a.intersects(b):
            area = a.intersection(b).area
            if area > 0.1:
                overlaps.append([str(polys["poly_id"].iloc[i]),
                                 str(polys["poly_id"].iloc[j]), float(area)])

    areas, distances = {}, {}
    for cid, g in polys.groupby("class_id"):
        areas[str(int(cid))] = {
            "poligonos": int(len(g)),
            "area_m2_media": float(g.geometry.area.mean()),
            "area_m2_min": float(g.geometry.area.min()),
            "area_m2_max": float(g.geometry.area.max()),
        }
        d = [a.distance(b) for a, b in combinations(g.geometry, 2)]
        distances[str(int(cid))] = float(min(d)) if d else None

    return {
        "epsg": int(polys.crs.to_epsg()) if polys.crs else None,
        "geometrias_invalidas": invalid,
        "solapes": overlaps,
        "areas_m2_por_clase": areas,
        "distancia_min_m_por_clase": distances,
    }


def main():
    df = pd.read_csv(TSV, sep="\t")
    polys = gpd.read_file(POLYGONS).to_crs(TARGET_EPSG)
    polys = polys[polys.geometry.notna() & ~polys.geometry.is_empty].copy()

    required = {"poly_id", "class_id", "class_name", "Latitude", "Longitude", *BANDS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {TSV.name}: {sorted(missing)}")
    if polys["poly_id"].duplicated().any():
        raise ValueError("Hay poly_id duplicados en el GPKG")
    unknown = set(df["poly_id"]) - set(polys["poly_id"])
    if unknown:
        raise ValueError(f"Píxeles con poly_id ausente del GPKG: {sorted(unknown)}")

    # Agregación espectral: mediana y rango intercuartílico por polígono.
    base = df.groupby("poly_id").agg(
        class_id=("class_id", "first"),
        class_name=("class_name", "first"),
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean"),
        n_pixels=("class_id", "size"),
    )
    med = df.groupby("poly_id")[BANDS].median()
    q1 = df.groupby("poly_id")[BANDS].quantile(0.25)
    q3 = df.groupby("poly_id")[BANDS].quantile(0.75)
    iqr = (q3 - q1).rename(columns={b: f"{b}_iqr" for b in BANDS})
    g = base.join(med).join(iqr)

    # Conteo de píxeles candidatos y descartados calculado por 01.
    sampling_path = RESULTS / "polygon_sampling.json"
    if not sampling_path.exists():
        raise FileNotFoundError(f"No existe {sampling_path}; ejecuta primero 01_extract_tsv.py")
    sampling = pd.DataFrame(json.loads(sampling_path.read_text(encoding="utf-8")))
    sampling = sampling.set_index("poly_id")[["n_candidates", "n_kept", "n_dropped"]]
    absent = set(g.index) - set(sampling.index)
    if absent:
        raise ValueError(f"Polígonos sin conteo de muestreo: {sorted(absent)}")
    g = g.join(sampling, how="left")
    if (g["n_pixels"] != g["n_kept"]).any():
        raise ValueError("n_pixels no coincide con n_kept en polygon_sampling.json")
    g = g.drop(columns="n_kept")

    # Índices sobre la mediana de las bandas.
    g = add_indices(g)
    g[BANDS] = g[BANDS].round(3)
    g[[f"{b}_iqr" for b in BANDS]] = g[[f"{b}_iqr" for b in BANDS]].round(3)
    g[INDICES] = g[INDICES].round(6)
    g[["Latitude", "Longitude"]] = g[["Latitude", "Longitude"]].round(7)
    g[["n_pixels", "n_candidates", "n_dropped"]] = g[
        ["n_pixels", "n_candidates", "n_dropped"]].astype(int)

    columns = (["poly_id", "class_id", "class_name", "Latitude", "Longitude",
                "n_pixels", "n_candidates", "n_dropped"] + BANDS
               + [f"{b}_iqr" for b in BANDS] + INDICES)
    out = (g.reset_index()[columns]
           .sort_values(["class_id", "poly_id"])
           .reset_index(drop=True))
    POLY_TSV.parent.mkdir(exist_ok=True)
    out.to_csv(POLY_TSV, sep="\t", index=False)

    # Banderas de calidad.
    outliers = pd.DataFrame({
        b: df.groupby("poly_id")[b].apply(outlier_fraction) for b in BANDS
    }).max(axis=1)
    dropped_frac = out["n_dropped"] / (out["n_pixels"] + out["n_dropped"]).replace(0, np.nan)
    flags_pixels = out.loc[out["n_pixels"] < MIN_PIXELS, "poly_id"].tolist()
    flags_dropped = out.loc[dropped_frac > MAX_DROPPED_FRAC, "poly_id"].tolist()
    flags_outliers = [str(p) for p in outliers[outliers > MAX_OUTLIER_FRAC].index]

    qc = {
        "dataset": POLY_TSV.name,
        "filas": int(len(out)),
        "columnas": int(out.shape[1]),
        "pixeles_totales": int(out["n_pixels"].sum()),
        "poligonos_por_clase": {str(int(k)): int(v) for k, v in
                                out.groupby("class_id").size().items()},
        "pixeles_por_clase": {str(int(k)): int(v) for k, v in
                              out.groupby("class_id")["n_pixels"].sum().items()},
        "nulos": int(out.isna().sum().sum()),
        "poly_id_duplicados": int(out["poly_id"].duplicated().sum()),
        "poligonos_sin_pixeles": [str(p) for p in
                                  sorted(set(polys["poly_id"]) - set(out["poly_id"]))],
        "muestreo": {
            "total_descartados": int(out["n_dropped"].sum()),
            "poligonos_con_menos_de_5_pixeles": [str(p) for p in flags_pixels],
            "poligonos_con_mas_de_20pct_descartado": [str(p) for p in flags_dropped],
            "poligonos_con_mas_de_30pct_atipicos": flags_outliers,
        },
        "geometria": polygon_geometry_qc(polys),
        "indices": {i: {"min": float(out[i].min()), "max": float(out[i].max())}
                    for i in INDICES},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "polygon_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset por polígono: {POLY_TSV}  filas={len(out)}  "
          f"columnas={out.shape[1]}  píxeles={qc['pixeles_totales']}")
    print(f"QC: {RESULTS / 'polygon_qc.json'}  "
          f"sin_pixeles={len(qc['poligonos_sin_pixeles'])}  "
          f"pocos_pixeles={len(flags_pixels)}  "
          f"descartado>20%={len(flags_dropped)}  atipicos>30%={len(flags_outliers)}")
    print(out.groupby(["class_id", "class_name"]).agg(
        poligonos=("poly_id", "size"), pixeles=("n_pixels", "sum")))


if __name__ == "__main__":
    main()
