@echo off
cd /d "%~dp0"
echo Starting FazDane Research Application on http://localhost:8501 ...
".python\Python312\python.exe" -m streamlit run app.py --server.port 8501
pause
