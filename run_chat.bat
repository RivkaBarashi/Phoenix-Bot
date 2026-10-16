@echo off
rem Starts the Phoenix chat UI. Double-click this file; the browser opens at http://localhost:8501
rem To stop: close this window (or press Ctrl+C in it).
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py
pause
