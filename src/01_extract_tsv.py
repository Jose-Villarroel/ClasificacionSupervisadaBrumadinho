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
from osgeo import gdal, ogr, osr
from pyproj import Transformer

from common import (BANDS, POLYGONS, SCL, SCL_INVALID, STACK,
                    TARGET_EPSG, TSV)

gdal.UseExceptions()


def main():
    src = POLYGONS
    print(f"Polígonos: {src.name}")
    polys = gpd.read_file(src).to_crs(TARGET_EPSG).reset_index(drop=True)
    polys = polys[polys.geometry.notna() & ~polys.geometry.is_empty]

    ds = gdal.Open(str(STACK))
    gt, w, h = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize
    stack = ds.ReadAsArray()                      # (12, h, w) uint16
    scl = gdal.Open(str(SCL)).ReadAsArray()

    # Rasterizar el índice de cada polígono (1..n); píxel incluido si su
    # centro cae dentro del polígono (ALL_TOUCHED=FALSE).
    srs = osr.SpatialReference(); srs.ImportFromEPSG(TARGET_EPSG)
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(gt); mem.SetProjection(srs.ExportToWkt())
    vds = ogr.GetDriverByName("MEM").CreateDataSource("")
    lyr = vds.CreateLayer("p", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("idx", ogr.OFTInteger))
    for i, g in enumerate(polys.geometry, start=1):
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetField("idx", i)
        f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
        lyr.CreateFeature(f)
    gdal.RasterizeLayer(mem, [1], lyr, options=["ATTRIBUTE=idx", "ALL_TOUCHED=FALSE"])
    idx = mem.ReadAsArray()

    rows, cols = np.nonzero(idx)
    keep = ~np.isin(scl[rows, cols], SCL_INVALID) & (stack[:, rows, cols] > 0).all(0)
    dropped = int((~keep).sum())
    rows, cols = rows[keep], cols[keep]
    pid = idx[rows, cols] - 1

    x = gt[0] + gt[1] * (cols + 0.5)
    y = gt[3] + gt[5] * (rows + 0.5)
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
