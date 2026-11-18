@echo off
rem Ejecuta un script con el Python que trae QGIS (GDAL, geopandas, pandas, matplotlib).
rem Uso:  run.bat src\01_extract_tsv.py
rem Si QGIS esta instalado en otra ruta o version, cambia la linea siguiente.
call "C:\Program Files\QGIS 3.44.13\bin\o4w_env.bat"
python %*
