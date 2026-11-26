@echo off
rem Alias de run.bat: todo el pipeline usa el mismo entorno.
rem Uso:  run_ml.bat src\03_train_models.py
call "%~dp0run.bat" %*
