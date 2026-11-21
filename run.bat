@echo off
rem Ejecuta un script con el entorno del proyecto (.venv si existe).
rem Crear el entorno una sola vez:
rem     python -m venv .venv
rem     .venv\Scripts\pip install -r requirements.txt
rem Uso:  run.bat src\01_extract_tsv.py
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" %*
) else (
    python %*
)
