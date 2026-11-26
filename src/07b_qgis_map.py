"""Fase 3: crea el proyecto QGIS y exporta el mapa temático.

Se ejecuta con el Python de QGIS:
    C:\\Users\\juand\\AppData\\Local\\Programs\\OSGeo4W\\bin\\python-qgis.bat src\\07b_qgis_map.py

Genera qgis/ProyectoFinal_Fase3.qgz y el mapa en report_fase3/figures.
"""

from pathlib import Path

from qgis.core import (Qgis, QgsApplication, QgsCoordinateReferenceSystem,
                       QgsFillSymbol, QgsLayoutExporter, QgsLayoutItemLabel,
                       QgsLayoutItemLegend, QgsLayoutItemMap,
                       QgsLayoutItemScaleBar, QgsLayoutPoint, QgsLayoutSize,
                       QgsLegendStyle, QgsPrintLayout, QgsProject,
                       QgsRasterLayer, QgsUnitTypes, QgsVectorLayer)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QFont

ROOT = Path(__file__).resolve().parents[1]
RASTER = ROOT / "data" / "prediction" / "landcover_lr.tif"
STYLE = ROOT / "data" / "prediction" / "landcover_lr.qml"
STUDY = ROOT / "data" / "study_area.gpkg"
ROI = ROOT / "data" / "ROI_Brumadinho.gpkg"
PROJECT = ROOT / "qgis" / "ProyectoFinal_Fase3.qgz"
OUT_PNG = ROOT / "report_fase3" / "figures" / "mapa_tematico_qgis.png"
OUT_PDF = ROOT / "report_fase3" / "figures" / "mapa_tematico_qgis.pdf"


def outline_layer(path, name, color, width, style="solid"):
    layer = QgsVectorLayer(str(path), name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"No se pudo abrir {path}")
    symbol = QgsFillSymbol.createSimple({
        "color": "0,0,0,0",
        "outline_color": color,
        "outline_width": str(width),
        "outline_style": style,
    })
    layer.renderer().setSymbol(symbol)
    return layer


def main():
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:32723"))
    project.setFilePathStorage(Qgis.FilePathType.Relative)

    raster = QgsRasterLayer(str(RASTER), "Cobertura predicha")
    if not raster.isValid():
        raise RuntimeError(f"No se pudo abrir {RASTER}")
    message, loaded = raster.loadNamedStyle(str(STYLE))
    if not loaded:
        raise RuntimeError(f"No se pudo aplicar el estilo: {message}")
    project.addMapLayer(raster)

    study = outline_layer(STUDY, "Área de estudio", "0,0,0,255", 0.6)
    roi = outline_layer(ROI, "Corredor del relave", "241,196,15,255", 0.7, "dash")
    project.addMapLayer(study)
    project.addMapLayer(roi)

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("Mapa temático Fase 3")
    page = layout.pageCollection().pages()[0]
    page.setPageSize(QgsLayoutSize(297, 150, QgsUnitTypes.LayoutMillimeters))

    title = QgsLayoutItemLabel(layout)
    title.setText("Coberturas predichas en Brumadinho")
    title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
    title.adjustSizeToText()
    title.attemptMove(QgsLayoutPoint(10, 8, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(title)

    subtitle = QgsLayoutItemLabel(layout)
    subtitle.setText("Sentinel-2 L2A, 1 de febrero de 2019 · modelo de regresión logística")
    subtitle.setFont(QFont("Arial", 10))
    subtitle.adjustSizeToText()
    subtitle.attemptMove(QgsLayoutPoint(10, 18, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(subtitle)

    map_item = QgsLayoutItemMap(layout)
    map_item.setFrameEnabled(True)
    map_item.setLayers([study, roi, raster])
    map_item.setCrs(QgsCoordinateReferenceSystem("EPSG:32723"))
    layout.addLayoutItem(map_item)
    map_item.attemptMove(QgsLayoutPoint(10, 30, QgsUnitTypes.LayoutMillimeters))
    map_item.attemptResize(QgsLayoutSize(200, 86, QgsUnitTypes.LayoutMillimeters))
    map_item.zoomToExtent(raster.extent())

    legend = QgsLayoutItemLegend(layout)
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(True)
    legend.setTitle("Coberturas")
    legend.setColumnCount(1)
    legend.setStyleFont(
        QgsLegendStyle.Title, QFont("Arial", 9, QFont.Weight.Bold)
    )
    legend.setStyleFont(QgsLegendStyle.SymbolLabel, QFont("Arial", 7))
    layout.addLayoutItem(legend)
    legend.attemptMove(QgsLayoutPoint(215, 32, QgsUnitTypes.LayoutMillimeters))
    legend.attemptResize(QgsLayoutSize(72, 100, QgsUnitTypes.LayoutMillimeters))

    scale = QgsLayoutItemScaleBar(layout)
    scale.setStyle("Single Box")
    scale.setLinkedMap(map_item)
    scale.applyDefaultSize()
    scale.setUnits(QgsUnitTypes.DistanceMeters)
    layout.addLayoutItem(scale)
    scale.attemptMove(QgsLayoutPoint(14, 130, QgsUnitTypes.LayoutMillimeters))

    project.layoutManager().addLayout(layout)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    PROJECT.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_PNG, OUT_PDF, PROJECT):
        path.unlink(missing_ok=True)
    project.write(str(PROJECT))

    exporter = QgsLayoutExporter(layout)
    image = exporter.renderPageToImage(0, QSize(), 300)
    if image.isNull():
        raise RuntimeError("No se pudo renderizar el PNG del layout")
    if not image.save(str(OUT_PNG), "PNG"):
        raise RuntimeError(f"No se pudo guardar {OUT_PNG}")

    pdf_settings = QgsLayoutExporter.PdfExportSettings()
    pdf_settings.dpi = 300
    result = exporter.exportToPdf(str(OUT_PDF), pdf_settings)
    if result != QgsLayoutExporter.Success:
        raise RuntimeError(f"Error exportando PDF: {result}")

    print("Proyecto:", PROJECT)
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)
    app.exitQgis()


if __name__ == "__main__":
    main()
