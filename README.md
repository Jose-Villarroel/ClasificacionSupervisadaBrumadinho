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

[`data/training_polygons.gpkg`](data/training_polygons.gpkg) contiene **77 polígonos**: 12 por clase y 17 de suelo expuesto, con los campos `class_id`, `class_name`, `poly_id` y `source`.

Se retiraron `5_03` y `5_11` por ambigüedad y se añadieron `5_13`–`5_19` como candidatos espectrales (`source = candidato`) sobre superficies desnudas homogéneas.

Los polígonos se propusieron con reglas de índices espectrales (NDVI, NDWI, MNDWI, NDBI, BSI y el ΔNDVI del midterm) y se revisan visualmente en QGIS. Las reglas exactas de cada clase están en el reporte, sección *Polígonos de entrenamiento*.

Para editarlos, abre el proyecto de QGIS (sección 6), activa la edición de la capa *Polígonos de entrenamiento*, haz los cambios y guarda. Después vuelve a correr los scripts (sección 7).

## 5. El dataset: `data/training_dataset.tsv`

Es una tabla separada por **tabulaciones**, con una fila por píxel: **1541 filas** y 17 columnas.

```
Latitude  Longitude  B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12  class_id  class_name  poly_id
```

| Clase | Polígonos | Píxeles |
|---|---|---|
| Vegetación densa | 12 | 300 |
| Vegetación baja | 12 | 300 |
| Agua | 12 | 108 |
| Relave minero | 12 | 300 |
| Suelo expuesto | 17 | 425 |
| Urbano | 12 | 108 |
| **Total** | **77** | **1541** |

### Dataset auxiliar por polígono

`data/training_polygons.tsv` es un derivado con **una fila por polígono**: mediana e IQR de cada banda, número de píxeles y los índices NDVI, NDWI, MNDWI, NDBI y BSI. Se usa para el experimento por polígono de la Fase 2; el entregable de la Fase 1 sigue siendo el TSV por píxel.

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
run.bat src\01b_polygon_dataset.py
run.bat src\02_figures.py
```

| Script | Qué hace | Entradas | Salidas |
|---|---|---|---|
| `src/common.py` | Rutas, bandas, clases, colores e índices compartidos | — | — |
| `src/01_extract_tsv.py` | Toma cada píxel cuyo centro cae dentro de un polígono, descarta los inválidos según la SCL, convierte las coordenadas a lat/lon y escribe el TSV | stack, SCL, polígonos | `data/training_dataset.tsv`, `results/polygon_sampling.json` |
| `src/01b_polygon_dataset.py` | Agrega cada polígono (mediana, IQR, índices) y hace control de calidad | TSV de píxeles, polígonos, muestreo | `data/training_polygons.tsv`, `results/polygon_qc.json` |
| `src/02_figures.py` | Mapas, firmas espectrales y tablas LaTeX del reporte | stack, SCL, polígonos, TSV, metadatos | `report/figures/`, `report/tables/` |

Si cambian los polígonos, corran los scripts en orden (01 → 01b → 02 → 03 → 04) y luego actualicen los reportes.

### Fase 2: entrenamiento

El mismo entorno `.venv` incluye scikit-learn. Desde la carpeta del repositorio:

```bat
run.bat src\03_train_models.py
run.bat src\04_phase2_figures.py
```

[`run_ml.bat`](run_ml.bat) se conserva como alias de `run.bat`.

| Script | Qué hace | Salidas |
|---|---|---|
| `src/03_train_models.py` | Separa entrenamiento y prueba **por polígono** (8 y 4 por clase, semilla 42). Pista A: entrena con los píxeles y evalúa por píxel y por polígono. Pista B: entrena con la mediana por polígono más índices. Reporta LOO por polígono, IC bootstrap e ICC | `results/` |
| `src/04_phase2_figures.py` | Matrices de confusión, gráfica de métricas, importancia de bandas, mapa de la división y tablas LaTeX | `report_fase2/figures/`, `report_fase2/tables/` |

Archivos en `results/`:

| Archivo | Contenido |
|---|---|
| `split.csv` | Cada polígono y si quedó en entrenamiento o prueba |
| `metrics_summary.csv` | Una fila por modelo: OA, Kappa, métricas macro, validación cruzada y parámetros |
| `confusion_<modelo>.csv` | Matriz de confusión (filas: clase real; columnas: predicha) |
| `per_class_<modelo>.csv` | Precisión, recall y F1 por clase |
| `feature_importance.csv` | Importancia de las bandas en DT y RF (pista A) |
| `tsv_quality.json` | Control de calidad del TSV |
| `polygon_sampling.json` | Píxeles candidatos, conservados y descartados por polígono |
| `polygon_qc.json` | Control de calidad del dataset auxiliar y de las geometrías |
| `metrics_polygon.csv` | Pista A evaluada por polígono (voto mayoritario y pureza) |
| `metrics_polytrain.csv` | Pista B (mediana por polígono) |
| `pixel_vs_polygon.csv` | Comparación directa de las tres evaluaciones |
| `bootstrap_ci.csv` | IC bootstrap al 95 % de OA, Kappa y F1 |
| `cv_folds.csv` | OA de cada pliegue LeaveOneGroupOut |
| `icc.csv` | ICC por banda y tamaño efectivo de muestra |
| `confusion_polygon_<modelo>.csv` | Matriz de la pista A por polígono |
| `confusion_polytrain_<modelo>.csv` | Matriz de la pista B |
| `per_class_polygon_<modelo>.csv` | Métricas por clase de la pista A por polígono |
| `per_class_polytrain_<modelo>.csv` | Métricas por clase de la pista B |
| `feature_importance_polytrain.csv` | Importancia de bandas e índices en DT y RF (pista B) |

### Resultados iniciales (Fase 2)

Métricas sobre el conjunto de prueba (24 polígonos, 472 píxeles). La OA de validación cruzada es *out-of-fold* con `LeaveOneGroupOut`; el ± es el error estándar de la OA por pliegue.

| Modelo | OA | Kappa | F1 macro | OA LOO |
|---|---|---|---|---|
| LR (línea base) | 0,934 | 0,919 | 0,938 | 0,932 ± 0,026 |
| DT | 0,905 | 0,881 | 0,888 | 0,894 ± 0,034 |
| RF (bonus) | 0,939 | 0,924 | 0,941 | 0,917 ± 0,028 |
| SVM | 0,947 | 0,935 | 0,958 | **0,942 ± 0,023** |
| **ANN** | **0,972** | **0,966** | **0,978** | 0,931 ± 0,023 |
| KNN | 0,939 | 0,924 | 0,945 | 0,923 ± 0,026 |

Al agregar las predicciones por polígono (pista A) y al entrenar con medianas (pista B):

| Modelo | A: OA polígono | A: Kappa | A: pureza | B: OA polígono | B: Kappa |
|---|---|---|---|---|---|
| LR | 0,917 | 0,900 | 0,986 | 0,958 | 0,950 |
| DT | 0,875 | 0,850 | 0,987 | 0,958 | 0,950 |
| RF | 0,958 | 0,950 | 0,969 | 0,958 | 0,950 |
| SVM | 0,958 | 0,950 | **1,000** | 0,917 | 0,900 |
| **ANN** | **1,000** | **1,000** | 0,978 | 0,958 | 0,950 |
| KNN | 0,958 | 0,950 | 0,978 | 0,917 | 0,900 |

En la pista B hay empate en 0,958/0,950 entre LR, DT, RF y ANN.

El ICC por banda (0,947–0,996) indica que los 1541 píxeles equivalen a unas 77–81 muestras independientes. Las diferencias entre modelos son de uno o dos polígonos sobre 24, así que no son concluyentes; por eso se reportan IC bootstrap.

La mejora frente a la versión anterior combina dos cambios de datos: se retiraron dos polígonos ambiguos y se añadieron siete candidatos de suelo. No es solo un cambio de modelo.

Vegetación y agua se clasifican sin errores. La SVM aún asigna completo `5_08` (suelo) a relave; la ANN lo corrige, pero asigna a suelo 25 píxeles de relave. Los antiguos `5_03` y `5_11` ya no están en el dataset.

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
│   ├── training_dataset.tsv       # dataset de entrenamiento (fila = píxel)
│   └── training_polygons.tsv      # dataset auxiliar (fila = polígono)
├── qgis/
│   └── ProyectoFinal.qgz          # proyecto de QGIS con todas las capas
├── src/
│   ├── common.py
│   ├── 01_extract_tsv.py
│   ├── 01b_polygon_dataset.py
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
