"""Fase 1 - Creación del dataset de entrenamiento (TSV).

Extrae, para cada píxel cuyo centro cae dentro de un polígono de entrenamiento,
sus coordenadas (WGS84) y los valores de las 12 bandas del stack Sentinel-2.
Una fila = un píxel.  Reflectancia en DN (reflectancia x 10 000).

Entradas: data/raw/S2_20190201_stack.tif, data/raw/S2_20190201_SCL.tif y data/training_polygons.gpkg.

Uso:  run.bat src\\01_extract_tsv.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize

from common import (BANDS, POLYGONS, SCL, SCL_INVALID, STACK,
                    TARGET_EPSG, TSV)


def main():
    src = POLYGONS
    print(f"Polígonos: {src.name}")
    polys = gpd.read_file(src).to_crs(TARGET_EPSG).reset_index(drop=True)
    polys = polys[polys.geometry.notna() & ~polys.geometry.is_empty]

    with rasterio.open(STACK) as ds:
        transform, w, h = ds.transform, ds.width, ds.height
        stack = ds.read()                          # (12, h, w) uint16
    with rasterio.open(SCL) as ds:
        scl = ds.read(1)

    # Rasterizar el índice de cada polígono (1..n); píxel incluido si su
    # centro cae dentro del polígono (all_touched=False).
    idx = rasterize(
        ((geom, i) for i, geom in enumerate(polys.geometry, start=1)),
        out_shape=(h, w), transform=transform, fill=0, dtype="int32",
        all_touched=False,
    )

    rows, cols = np.nonzero(idx)
    keep = ~np.isin(scl[rows, cols], SCL_INVALID) & (stack[:, rows, cols] > 0).all(0)
    dropped = int((~keep).sum())
    rows, cols = rows[keep], cols[keep]
    pid = idx[rows, cols] - 1

    x = transform.c + transform.a * (cols + 0.5) + transform.b * (rows + 0.5)
    y = transform.f + transform.d * (cols + 0.5) + transform.e * (rows + 0.5)
    lon, lat = Transformer.from_crs(TARGET_EPSG, 4326, always_xy=True).transform(x, y)

    df = pd.DataFrame({"Latitude": np.round(lat, 7), "Longitude": np.round(lon, 7)})
    for i, b in enumerate(BANDS):
        df[b] = stack[i, rows, cols].astype(int)
    df["class_id"] = polys["class_id"].to_numpy()[pid].astype(int)
    df["class_name"] = polys["class_name"].to_numpy()[pid]
    df["poly_id"] = polys["poly_id"].to_numpy()[pid]
    df = df.sort_values(["class_id", "poly_id", "Latitude", "Longitude"],
                        ascending=[True, True, False, True])
    df.to_csv(TSV, sep="\t", index=False)

    print(f"TSV: {TSV}  filas={len(df)}  columnas={df.shape[1]}  "
          f"píxeles descartados (SCL/nodata)={dropped}")
    print(df.groupby(["class_id", "class_name"]).agg(
        poligonos=("poly_id", "nunique"), pixeles=("poly_id", "size")))


if __name__ == "__main__":
    main()
