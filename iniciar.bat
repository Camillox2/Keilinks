@echo off
echo Iniciando Keilinks...
cd /d "%~dp0"
if exist ".venv-unsloth\Scripts\activate.bat" (
  call .venv-unsloth\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
  call venv\Scripts\activate.bat
)

REM Abre o navegador APOS 2 segundos (servidor precisa subir antes)
timeout /t 2 /nobreak >nul
start "" "http://localhost:5000"

python -m api.servidor_v4
