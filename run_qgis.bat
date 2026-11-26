@echo off
rem Ejecuta un script con el Python de QGIS.
rem Uso:  run_qgis.bat src\07b_qgis_map.py
set "QGIS_PY=%LOCALAPPDATA%\Programs\OSGeo4W\bin\python-qgis.bat"
if exist "%QGIS_PY%" (
    "%QGIS_PY%" %*
) else (
    python-qgis.bat %*
)
