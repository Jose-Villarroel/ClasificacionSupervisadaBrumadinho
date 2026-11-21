# Clasificación supervisada de coberturas en Brumadinho con Sentinel-2

**Proyecto final:** *A Comparative Analysis of Machine Learning Architectures for Satellite-Based Land Cover Classification*
Sistemas Digitales Emergentes, Pontificia Universidad Javeriana

El objetivo es comparar cinco modelos de *machine learning* para clasificar las coberturas terrestres de una escena Sentinel-2 y generar un mapa temático con el área de cada clase. Los modelos son regresión logística, árbol de decisión o Random Forest, SVM, red neuronal y KNN o Naive Bayes.

| Fase | Entrega | Estado |
|---|---|---|
| 1. Definición del problema | Evento, área de estudio, metadatos de la imagen y dataset (TSV) | ✅ Lista |
| 2. Preprocesamiento y entrenamiento | Documentación del TSV y métricas iniciales: matriz de confusión y exactitud global | ✅ Lista |
| 3. Síntesis y predicción | GeoTIFF temático, reporte técnico y código (ZIP) | ⏳ Pendiente |

---

## 1. Caso de estudio

- **Evento:** colapso de la presa I de la mina Córrego do Feijão, en Brumadinho (Minas Gerais, Brasil), el **25-01-2019**. El lodo de relaves cubrió el valle del Ribeirão Ferro-Carvão y llegó al río Paraopeba.
- **Escena:** una sola imagen, **posterior** al evento: **01-02-2019**. Es la misma fecha *t2* del midterm. En ella el lodo aparece como una cobertura propia que se puede clasificar.
- **Área de estudio:** rectángulo de ~74 km² que incluye la mina, el corredor del lodo, el casco urbano y el río.

| Parámetro | Valor |
|---|---|
| Longitud (WGS84) | −44,224111° a −44,096943° |
| Latitud (WGS84) | −20,159193° a −20,108763° |
| Sistema de referencia de trabajo | WGS 84 / UTM zona 23S (**EPSG:32723**) |
| Ráster | 1338 × 570 píxeles de 10 m |

## 2. La imagen satelital

| Campo | Valor |
|---|---|
| Producto | `S2A_MSIL2A_20190201T131241_N0212_R138_T23KNT_20201007T130407` |
| Satélite / nivel | Sentinel-2A, **L2A** (reflectancia de superficie, corregida atmosféricamente con Sen2Cor) |
| Tile / órbita | 23KNT / R138 |
| Nubosidad de la escena | 1,14 % (0,11 % de píxeles inválidos dentro del área) |
| Bandas | **12**: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 (B10 no existe en L2A) |
| Resolución | Todas remuestreadas a **10 m**. Las de 20 y 60 m, con interpolación bilineal |
| Valores | UInt16, **reflectancia × 10 000**. El *processing baseline* 02.12 no tiene offset |

Todos los metadatos completos están en [`data/raw/metadata.json`](data/raw/metadata.json).

### Cómo se obtuvo

La imagen ya viene incluida en `data/raw/`, así que **no hay que descargarla**. Si alguien necesita reconstruirla, se puede hacer a mano:

1. Entra a [Copernicus Browser](https://browser.dataspace.copernicus.eu/), busca la fecha **2019-02-01** con Sentinel-2 L2A y dibuja el rectángulo del área de estudio.
2. En la pestaña **Analytical** descarga las 12 bandas y la **SCL**, en formato **Raw**, TIFF de 16 bits.
3. En QGIS:
   - Ve a *Ráster → Miscelánea → Construir ráster virtual*, marca **"Colocar cada archivo de entrada en una banda separada"** y respeta el orden B01…B12.
   - Usa *Ráster → Proyecciones → Combar (reproyectar)* hacia EPSG:32723 con tamaño de píxel de 10 m.
   - Exporta el resultado como `S2_20190201_stack.tif`.

La versión incluida se descargó del catálogo público de [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) (colección `sentinel-2-l2a`). Es la misma adquisición.

## 3. Clases de cobertura

| `class_id` | `class_name` (en el TSV) | Qué es |
|---|---|---|
| 1 | Vegetacion densa | Bosque |
| 2 | Vegetacion baja | Pasto, cultivos |
| 3 | Agua | Lagunas, río |
| 4 | Relave minero | Lodo del colapso |
| 5 | Suelo expuesto | Tajos de mina, suelo desnudo |
| 6 | Urbano | Zonas construidas |

El `class_id` será el valor de cada píxel en el GeoTIFF temático de la Fase 3.

## 4. Polígonos de entrenamiento

[`data/training_polygons.gpkg`](data/training_polygons.gpkg) contiene **72 polígonos, 12 por clase**, con los campos `class_id`, `class_name` y `poly_id`.

Los polígonos se propusieron con reglas de índices espectrales (NDVI, NDWI, MNDWI, NDBI, BSI y el ΔNDVI del midterm) y se revisan visualmente en QGIS. Las reglas exactas de cada clase están en el reporte, sección *Polígonos de entrenamiento*.

Para editarlos, abre el proyecto de QGIS (sección 6), activa la edición de la capa *Polígonos de entrenamiento*, haz los cambios y guarda. Después vuelve a correr los scripts (sección 7).

## 5. El dataset: `data/training_dataset.tsv`

Es una tabla separada por **tabulaciones**, con una fila por píxel: **1416 filas** y 17 columnas.

```
Latitude  Longitude  B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12  class_id  class_name  poly_id
```

| Clase | Polígonos | Píxeles |
|---|---|---|
| Vegetación densa | 12 | 300 |
| Vegetación baja | 12 | 300 |
| Agua | 12 | 108 |
| Relave minero | 12 | 300 |
| Suelo expuesto | 12 | 300 |
| Urbano | 12 | 108 |
| **Total** | **72** | **1416** |

Para leerlo en Python:

```python
import pandas as pd
df = pd.read_csv("data/training_dataset.tsv", sep="\t")
X = df[["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]] / 10000  # reflectancia
y = df["class_id"]
groups = df["poly_id"]   # para separar entrenamiento y prueba POR POLÍGONO
```

> **Importante para la Fase 2:** separen entrenamiento y prueba **por polígono** usando `poly_id`, por ejemplo con `GroupShuffleSplit` o `GroupKFold` de scikit-learn. Si mezclan píxeles de un mismo cuadrito en ambos conjuntos, las métricas salen infladas.

## 6. Proyecto de QGIS

Abre [`qgis/ProyectoFinal.qgz`](qgis/ProyectoFinal.qgz) con QGIS 3.x. Trae estas capas, ya simbolizadas:

- Polígonos de entrenamiento, con un color por clase;
- ROI del corredor de relaves (midterm) y área de estudio;
- color verdadero (B04-B03-B02), falso color infrarrojo (B08-B04-B03) y falso color SWIR (B12-B08-B04);
- SCL, la capa de nubes y calidad.

Las rutas son relativas: funciona en cualquier computador siempre que se mantenga la estructura de carpetas. Si abre lejos de las capas, haz clic derecho en una capa → **Zoom a la capa**.

## 7. Código

Todo el pipeline usa un **único entorno de Python 3.10+** y no requiere QGIS para ejecutarse. Se crea una sola vez:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

[`run.bat`](run.bat) usa ese entorno (o el `python` del sistema si `.venv` no existe). Desde la carpeta del repositorio:

```bat
run.bat src\01_extract_tsv.py
run.bat src\02_figures.py
```

| Script | Qué hace | Entradas | Salidas |
|---|---|---|---|
| `src/common.py` | Rutas, bandas, clases y colores compartidos | — | — |
| `src/01_extract_tsv.py` | Toma cada píxel cuyo centro cae dentro de un polígono, descarta los inválidos según la SCL, convierte las coordenadas a lat/lon y escribe el TSV | stack, SCL, polígonos | `data/training_dataset.tsv` |
| `src/02_figures.py` | Mapas, firmas espectrales y tablas LaTeX del reporte | stack, SCL, polígonos, TSV, metadatos | `report/figures/`, `report/tables/` |

Si cambian los polígonos, corran los scripts en orden (01 → 02 → 03 → 04) y luego actualicen los reportes.

### Fase 2: entrenamiento

El mismo entorno `.venv` incluye scikit-learn. Desde la carpeta del repositorio:

```bat
run.bat src\03_train_models.py
run.bat src\04_phase2_figures.py
```

[`run_ml.bat`](run_ml.bat) se conserva como alias de `run.bat`.

| Script | Qué hace | Salidas |
|---|---|---|
| `src/03_train_models.py` | Separa entrenamiento y prueba **por polígono** (8 y 4 por clase, semilla 42), estandariza las bandas, entrena LR, DT, RF, SVM, ANN y KNN, y calcula la matriz de confusión, OA, Kappa, precisión, recall, F1 y una validación cruzada agrupada por polígono | `results/` |
| `src/04_phase2_figures.py` | Matrices de confusión, gráfica de métricas, importancia de bandas, mapa de la división y tablas LaTeX | `report_fase2/figures/`, `report_fase2/tables/` |

Archivos en `results/`:

| Archivo | Contenido |
|---|---|
| `split.csv` | Cada polígono y si quedó en entrenamiento o prueba |
| `metrics_summary.csv` | Una fila por modelo: OA, Kappa, métricas macro, validación cruzada y parámetros |
| `confusion_<modelo>.csv` | Matriz de confusión (filas: clase real; columnas: predicha) |
| `per_class_<modelo>.csv` | Precisión, recall y F1 por clase |
| `feature_importance.csv` | Importancia de las bandas en DT y RF |
| `tsv_quality.json` | Control de calidad del TSV |

### Resultados iniciales (Fase 2)

Métricas sobre el conjunto de prueba (24 polígonos, 472 píxeles):

| Modelo | OA | Kappa | F1 macro | OA validación cruzada |
|---|---|---|---|---|
| LR (línea base) | 0,883 | 0,857 | 0,868 | 0,891 ± 0,062 |
| DT | 0,788 | 0,741 | 0,792 | 0,803 ± 0,066 |
| RF (bonus) | 0,835 | 0,797 | 0,827 | 0,873 ± 0,047 |
| **SVM** | **0,894** | **0,870** | **0,883** | 0,906 ± 0,054 |
| ANN | 0,871 | 0,841 | 0,878 | **0,916 ± 0,052** |
| KNN | 0,828 | 0,789 | 0,828 | 0,888 ± 0,060 |

Vegetación y agua se clasifican sin errores. La confusión está entre suelo expuesto, relave y urbano: los polígonos de prueba `5_03` y `5_11` se clasifican completos como urbano y como relave, respectivamente.

## 8. Reportes (LaTeX)

- [`report/`](report/): documento de la Fase 1.
- [`report_fase2/`](report_fase2/): documento de la Fase 2.
- [`report_completo/`](report_completo/): documento único con las Fases 1 y 2 (copia de las figuras y tablas de ambas carpetas; si se regeneran, hay que copiarlas de nuevo).

Cada carpeta contiene `main.tex`, `references.bib`, las figuras y las tablas generadas.

Para compilar cualquiera de los dos en Overleaf:
1. Comprime la carpeta del reporte.
2. Ve a **New Project → Upload Project** y sube el ZIP.
3. Compila con pdfLaTeX. Overleaf corre biber automáticamente.

## 9. Estructura del repositorio

```
.
├── README.md
├── run.bat                        # ejecuta scripts con el entorno .venv (o python)
├── run_ml.bat                     # alias de run.bat
├── requirements.txt               # dependencias del entorno único
├── data/
│   ├── raw/
│   │   ├── S2_20190201_stack.tif  # imagen de 12 bandas, 10 m, EPSG:32723
│   │   ├── S2_20190201_SCL.tif    # clasificación de escena (nubes y calidad)
│   │   └── metadata.json          # metadatos de la escena
│   ├── study_area.gpkg            # rectángulo del área de estudio
│   ├── ROI_Brumadinho.gpkg        # corredor del relave (midterm)
│   ├── training_polygons.gpkg     # 72 polígonos de entrenamiento
│   └── training_dataset.tsv       # dataset de entrenamiento
├── qgis/
│   └── ProyectoFinal.qgz          # proyecto de QGIS con todas las capas
├── src/
│   ├── common.py
│   ├── 01_extract_tsv.py
│   ├── 02_figures.py
│   ├── 03_train_models.py         # Fase 2: entrenamiento y métricas
│   └── 04_phase2_figures.py       # Fase 2: figuras y tablas
├── results/                       # métricas y matrices de confusión (CSV)
├── report/                        # reporte de la Fase 1
├── report_fase2/                  # reporte de la Fase 2
└── report_completo/               # reporte unificado (Fases 1 y 2)
```

## 10. Próximos pasos

- **Fase 3:**
  - revisar en QGIS los polígonos de suelo expuesto `5_03` y `5_11`;
  - optimizar los hiperparámetros con validación cruzada agrupada por polígono;
  - comparar los modelos con Kappa, precisión, recall y F1;
  - aplicar el mejor modelo a `S2_20190201_stack.tif` y exportar el GeoTIFF temático conservando el CRS y el *transform*;
  - simbolizarlo en QGIS y calcular el área por clase.
