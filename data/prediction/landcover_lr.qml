<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40.0-Bratislava" styleCategories="AllStyleCategories">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <colorPalette>
    <paletteEntry value="0" color="#000000" alpha="0" label="Sin datos"/>
    <paletteEntry value="1" color="#1b7837" alpha="255" label="Vegetación densa"/>
    <paletteEntry value="2" color="#a6d96a" alpha="255" label="Vegetación baja"/>
    <paletteEntry value="3" color="#2166ac" alpha="255" label="Agua"/>
    <paletteEntry value="4" color="#8c510a" alpha="255" label="Relave minero"/>
    <paletteEntry value="5" color="#f4a582" alpha="255" label="Suelo expuesto"/>
    <paletteEntry value="6" color="#d6604d" alpha="255" label="Urbano"/>
      </colorPalette>
      <rasterTransparency/>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation grayscaleMode="0" saturation="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
  <layerGeometryType>2</layerGeometryType>
</qgis>
