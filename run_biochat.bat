@echo off
title BioChat Server - Tro ly Sinh hoc 12
echo =====================================================================
echo    DANG KHOI DONG BIOCHAT SERVER (http://127.0.0.1:8000)
echo =====================================================================
cd /d "%~dp0biochat_backend"
python app.py
pause
