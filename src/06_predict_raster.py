"""Fase 3: inferencia del modelo final sobre todo el ráster.

Genera el GeoTIFF temático, su estilo QGIS y la tabla de áreas por clase.
La inferencia se hace por bloques de filas para no cargar todo el stack.

Uso:
    run.bat src\\06_predict_raster.py
"""

import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.windows import Window

from common import (BANDS, CLASS_COLORS, CLASS_LABELS, CLASSES, FINAL_MODEL,
                    MODEL_CARD, PREDICTION, RESULTS, ROI, SCL, SCL_INVALID,
                    STUDY_AREA, STACK)

BLOCK_ROWS = 128
NODATA = 0


def hex_to_rgba(value, alpha=255):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def write_qml(path, model_key):
    entries = ['    <paletteEntry value="0" color="#000000" alpha="0" label="Sin datos"/>']
    for class_id, color in CLASS_COLORS.items():
        label = CLASS_LABELS[class_id].replace("&", "&amp;").replace('"', "&quot;")
        entries.append(
            f'    <paletteEntry value="{class_id}" color="{color}" '
            f'alpha="255" label="{label}"/>'
        )
    text = "\n".join([
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>",
        '<qgis version="3.40.0-Bratislava" styleCategories="AllStyleCategories">',
        "  <pipe>",
        "    <provider>",
        '      <resampling enabled="false" maxOversampling="2" '
        'zoomedInResamplingMethod="nearestNeighbour" '
        'zoomedOutResamplingMethod="nearestNeighbour"/>',
        "    </provider>",
        f'    <rasterrenderer type="paletted" band="1" opacity="1" '
        f'alphaBand="-1" nodataColor="">',
        "      <colorPalette>",
        *entries,
        "      </colorPalette>",
        "      <rasterTransparency/>",
        "    </rasterrenderer>",
        '    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>',
        '    <huesaturation grayscaleMode="0" saturation="0" colorizeStrength="100"/>',
        '    <rasterresampler maxOversampling="2"/>',
        "  </pipe>",
        "  <blendMode>0</blendMode>",
        f"  <layerGeometryType>2</layerGeometryType>",
        "</qgis>",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def valid_mask_for(src):
    study = gpd.read_file(STUDY_AREA).to_crs(src.crs)
    geometries = list(study.geometry)
    inside = geometry_mask(
        geometries,
        out_shape=(src.height, src.width),
        transform=src.transform,
        invert=True,
        all_touched=False,
    )
    with rasterio.open(SCL) as scl:
        if scl.shape != src.shape or scl.transform != src.transform:
            raise ValueError("La SCL no está alineada con el stack")
        scl_data = scl.read(1)
    invalid = np.isin(scl_data, SCL_INVALID)
    return inside & ~invalid


def main():
    artifact = joblib.load(FINAL_MODEL)
    model = artifact["model"]
    model_key = artifact["model_key"]
    features = list(artifact["features"])

    if features != BANDS:
        raise ValueError(f"Features del modelo no coinciden: {features}")
    if list(artifact["classes"]) != sorted(CLASSES):
        raise ValueError("Clases del modelo no coinciden con common.CLASSES")

    PREDICTION.mkdir(parents=True, exist_ok=True)
    output = PREDICTION / f"landcover_{model_key.lower()}.tif"
    style = PREDICTION / f"landcover_{model_key.lower()}.qml"

    with rasterio.open(STACK) as src:
        if src.count != len(BANDS):
            raise ValueError(f"El stack tiene {src.count} bandas, se esperaban {len(BANDS)}")
        valid_mask = valid_mask_for(src)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=NODATA,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
            description="Cobertura predicha Fase 3",
        )
        colormap = {NODATA: (0, 0, 0, 0)}
        for class_id, color in CLASS_COLORS.items():
            colormap[class_id] = hex_to_rgba(color)

        counts = np.zeros(len(CLASSES) + 1, dtype=np.int64)
        with rasterio.open(output, "w", **profile) as dst:
            dst.update_tags(model=model_key, features=",".join(BANDS))
            for row in range(0, src.height, BLOCK_ROWS):
                height = min(BLOCK_ROWS, src.height - row)
                window = Window(0, row, src.width, height)
                block = src.read(window=window).astype(np.float64) / 10000.0
                valid = valid_mask[row:row + height] & np.any(block != 0, axis=0)
                result = np.full((height, src.width), NODATA, dtype=np.uint8)
                if valid.any():
                    pixels = block[:, valid].T
                    result[valid] = model.predict(pixels).astype(np.uint8)
                dst.write(result, 1, window=window)
                counts += np.bincount(result.ravel(), minlength=len(counts))
            dst.write_colormap(1, colormap)

    write_qml(style, model_key)

    transform = src.transform
    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    rows = []
    valid_pixels = int(counts[1:].sum())
    for class_id in sorted(CLASSES):
        pixels = int(counts[class_id])
        area_m2 = pixels * pixel_area_m2
        rows.append({
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "pixels": pixels,
            "hectares": area_m2 / 10000.0,
            "km2": area_m2 / 1_000_000.0,
            "percent_valid": 100.0 * pixels / valid_pixels if valid_pixels else 0.0,
        })
    area = pd.DataFrame(rows)
    area.to_csv(RESULTS / "phase3_area_by_class.csv", index=False)

    roi = gpd.read_file(ROI).to_crs(src.crs)
    with rasterio.open(output) as predicted:
        corridor, _ = mask(
            predicted, [roi.geometry.iloc[0]], crop=False, all_touched=False
        )
    corridor_counts = np.bincount(corridor[0].ravel(), minlength=len(counts))
    corridor_rows = [
        {
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "pixels": int(corridor_counts[class_id]),
            "km2": int(corridor_counts[class_id]) * pixel_area_m2 / 1_000_000.0,
        }
        for class_id in CLASSES
    ]
    pd.DataFrame(corridor_rows).to_csv(
        RESULTS / "phase3_area_by_corridor.csv", index=False
    )

    no_data = int(counts[NODATA])
    summary = {
        "model": model_key,
        "raster": str(output),
        "style": str(style),
        "width": src.width,
        "height": src.height,
        "crs": str(src.crs),
        "transform": list(src.transform),
        "pixel_area_m2": pixel_area_m2,
        "class_pixels": {CLASSES[cid]: int(counts[cid]) for cid in CLASSES},
        "valid_pixels": valid_pixels,
        "no_data_pixels": no_data,
        "valid_area_km2": valid_pixels * pixel_area_m2 / 1_000_000.0,
        "no_data_area_km2": no_data * pixel_area_m2 / 1_000_000.0,
        "model_card": str(MODEL_CARD),
    }
    (RESULTS / "phase3_area_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Modelo: {model_key}")
    print(f"GeoTIFF: {output}")
    print(f"Estilo:  {style}")
    print(f"Área válida: {summary['valid_area_km2']:.4f} km²")
    print(area.to_string(index=False))


if __name__ == "__main__":
    main()
