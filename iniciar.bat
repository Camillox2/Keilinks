@echo off
echo Iniciando Keilinks...
cd /d D:\Keilinks
call venv\Scripts\activate

REM Abre o navegador APOS 2 segundos (servidor precisa subir antes)
timeout /t 2 /nobreak >nul
start "" "http://localhost:5000"

python api/servidor.py
