"""Constantes y rutas compartidas por los scripts del proyecto final."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"

# Fase 2
RESULTS = ROOT / "results"
FIG2 = ROOT / "report_fase2" / "figures"
TAB2 = ROOT / "report_fase2" / "tables"

# Área de estudio (WGS84): lon_min, lat_min, lon_max, lat_max
AOI_BBOX = (-44.224111, -20.159193, -44.096943, -20.108763)
TARGET_EPSG = 32723  # WGS 84 / UTM zona 23S

# 12 bandas de Sentinel-2 L2A (B10 no existe en L2A)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
         "B07", "B08", "B8A", "B09", "B11", "B12"]

# Índices espectrales calculados a partir de las bandas
INDICES = ["NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]

# Longitud de onda central (nm) y resolución nativa (m), Sentinel-2A
BAND_INFO = {
    "B01": ("Aerosol costero", 443, 60),
    "B02": ("Azul", 490, 10),
    "B03": ("Verde", 560, 10),
    "B04": ("Rojo", 665, 10),
    "B05": ("Borde rojo 1", 705, 20),
    "B06": ("Borde rojo 2", 740, 20),
    "B07": ("Borde rojo 3", 783, 20),
    "B08": ("NIR", 842, 10),
    "B8A": ("NIR estrecho", 865, 20),
    "B09": ("Vapor de agua", 945, 60),
    "B11": ("SWIR 1", 1610, 20),
    "B12": ("SWIR 2", 2190, 20),
}

STACK = RAW / "S2_20190201_stack.tif"
SCL = RAW / "S2_20190201_SCL.tif"
METADATA = RAW / "metadata.json"
STUDY_AREA = DATA / "study_area.gpkg"
ROI = DATA / "ROI_Brumadinho.gpkg"
POLYGONS = DATA / "training_polygons.gpkg"
TSV = DATA / "training_dataset.tsv"
POLY_TSV = DATA / "training_polygons.tsv"

# Clases SCL consideradas inválidas (sin datos, saturado, sombra, nubes, cirros)
SCL_INVALID = [0, 1, 3, 8, 9, 10]

CLASSES = {
    1: "Vegetacion densa",
    2: "Vegetacion baja",
    3: "Agua",
    4: "Relave minero",
    5: "Suelo expuesto",
    6: "Urbano",
}
# Nombres con tildes para figuras y tablas (el TSV usa CLASSES, sin tildes)
CLASS_LABELS = {
    1: "Vegetación densa",
    2: "Vegetación baja",
    3: "Agua",
    4: "Relave minero",
    5: "Suelo expuesto",
    6: "Urbano",
}
CLASS_COLORS = {
    1: "#1b7837",
    2: "#a6d96a",
    3: "#2166ac",
    4: "#8c510a",
    5: "#f4a582",
    6: "#d6604d",
}


def add_indices(df):
    """Añade NDVI, NDWI, MNDWI, NDBI y BSI a un DataFrame con las bandas BANDS."""
    b = {band: df[band].astype("float64") for band in BANDS}
    eps = 1e-6
    df["NDVI"] = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    df["NDWI"] = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps)
    df["MNDWI"] = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"] + eps)
    df["NDBI"] = (b["B11"] - b["B08"]) / (b["B11"] + b["B08"] + eps)
    df["BSI"] = ((b["B11"] + b["B04"]) - (b["B08"] + b["B02"])) / (
        (b["B11"] + b["B04"]) + (b["B08"] + b["B02"]) + eps)
    return df
