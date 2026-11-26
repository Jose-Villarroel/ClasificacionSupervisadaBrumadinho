"""Fase 3: validación de artefactos y consistencia espacial.

Comprueba el split, el modelo final, el GeoTIFF, las áreas y los archivos del
reporte. Escribe results/phase3_validation.json.

Uso:
    run.bat src\\08_validate_phase3.py
"""

import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from common import (BANDS, CLASSES, FINAL_MODEL, MODEL_CARD, PREDICTION,
                    RESULTS, ROI, SCL, SCL_INVALID, STUDY_AREA, STACK)

ROOT = Path(__file__).resolve().parents[1]
checks = []


def check(name, condition, detail=""):
    checks.append({"check": name, "ok": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def main():
    split = pd.read_csv(RESULTS / "split.csv")
    train_ids = set(split.loc[split["set"] == "entrenamiento", "poly_id"])
    test_ids = set(split.loc[split["set"] == "prueba", "poly_id"])
    check("split_train", len(train_ids) == 64, f"{len(train_ids)} polígonos")
    check("split_test", len(test_ids) == 24, f"{len(test_ids)} polígonos")
    check("split_disjunto", not (train_ids & test_ids), "sin solapamiento")

    artifact = joblib.load(FINAL_MODEL)
    check("modelo_carga", "model" in artifact and "features" in artifact)
    check("modelo_bandas", list(artifact["features"]) == BANDS)
    check("modelo_clases", list(artifact["classes"]) == sorted(CLASSES))
    card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
    check("modelo_card", card["selected_model"] == artifact["model_key"])
    check("modelo_entrenamiento", card["train"]["polygons"] == 88,
          f"{card['train']['polygons']} polígonos")

    metrics = pd.read_csv(RESULTS / "phase3_test_metrics.csv")
    check("metricas_filas", len(metrics) == 6, f"{len(metrics)} modelos")
    check("metricas_final", int(metrics["selected_by_cv"].sum()) == 1)
    check("metricas_sin_nan", not metrics.isna().any().any())

    with rasterio.open(STACK) as stack:
        with rasterio.open(SCL) as scl:
            check("scl_grilla", scl.shape == stack.shape and scl.transform == stack.transform)
            scl_data = scl.read(1)
        with rasterio.open(PREDICTION / "landcover_lr.tif") as pred:
            check("raster_grilla", pred.shape == stack.shape)
            check("raster_crs", pred.crs == stack.crs, str(pred.crs))
            check("raster_transform", pred.transform == stack.transform)
            check("raster_dtype", pred.dtypes == ("uint8",), str(pred.dtypes))
            check("raster_nodata", pred.nodata == 0, str(pred.nodata))
            array = pred.read(1)
        stack_data = stack.read()
        study = gpd.read_file(STUDY_AREA).to_crs(stack.crs)
        inside = geometry_mask(
            list(study.geometry), out_shape=stack.shape,
            transform=stack.transform, invert=True, all_touched=False
        )
        expected = inside & ~np.isin(scl_data, SCL_INVALID) & np.any(stack_data != 0, axis=0)
        check("raster_fuera_estudio", not np.any(array[~inside] != 0),
              f"{int(np.count_nonzero(array[~inside]))} píxeles fuera")
        check("raster_scl", not np.any(array[np.isin(scl_data, SCL_INVALID)] != 0))
        check("raster_clases", set(np.unique(array)).issubset(set(range(7))),
              str(np.unique(array).tolist()))
        check("raster_mascara", np.array_equal(array > 0, expected),
              f"{int(np.count_nonzero((array > 0) != expected))} discrepancias")
        check("raster_validos", np.all((array[expected] >= 1) & (array[expected] <= 6)))

    area = pd.read_csv(RESULTS / "phase3_area_by_class.csv")
    summary = json.loads((RESULTS / "phase3_area_summary.json").read_text(encoding="utf-8"))
    counts = np.bincount(array.ravel(), minlength=7)
    check("area_filas", len(area) == 6)
    check("area_pixeles", all(int(row.pixels) == int(counts[int(row.class_id)])
                              for row in area.itertuples()))
    check("area_total", int(area["pixels"].sum()) == summary["valid_pixels"])
    check("area_km2", abs(float(area["km2"].sum()) - summary["valid_area_km2"]) < 1e-9)
    check("area_valida", 73.9 < summary["valid_area_km2"] < 74.3,
          f"{summary['valid_area_km2']:.3f} km²")

    corridor = pd.read_csv(RESULTS / "phase3_area_by_corridor.csv")
    check("corredor_filas", len(corridor) == 6)
    check("corredor_total", int(corridor["pixels"].sum()) <= summary["valid_pixels"])

    required = [
        PREDICTION / "landcover_lr.qml",
        RESULTS / "phase3_permutation_importance.csv",
        ROOT / "qgis" / "ProyectoFinal_Fase3.qgz",
        ROOT / "report_fase3" / "figures" / "mapa_tematico_qgis.png",
        ROOT / "report_fase3" / "main.pdf",
    ]
    for path in required:
        check(f"existe_{path.name}", path.exists(), str(path))

    output = {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "model": artifact["model_key"],
        "test_metrics": metrics.loc[metrics["selected_by_cv"]].iloc[0].to_dict(),
        "valid_pixels": summary["valid_pixels"],
        "valid_area_km2": summary["valid_area_km2"],
        "raster": str(PREDICTION / "landcover_lr.tif"),
    }
    (RESULTS / "phase3_validation.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8"
    )
    print(f"Validación OK: {len(checks)} comprobaciones")
    print(f"Área válida: {summary['valid_area_km2']:.3f} km²")
    print(f"Resultado: {RESULTS / 'phase3_validation.json'}")


if __name__ == "__main__":
    main()
