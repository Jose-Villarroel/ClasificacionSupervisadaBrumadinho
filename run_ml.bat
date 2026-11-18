@echo off
rem Ejecuta un script con el entorno de machine learning (.venv, scikit-learn).
rem Crear el entorno una sola vez:
rem     python -m venv .venv
rem     .venv\Scripts\pip install -r requirements.txt
rem Uso:  run_ml.bat src\03_train_models.py
"%~dp0.venv\Scripts\python.exe" %*
