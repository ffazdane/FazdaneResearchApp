@echo off
cd /d "%~dp0"
echo Installing missing packages into bundled Python environment...
".python\Python312\python.exe" -m pip install scikit-learn st-img-pastebutton
echo Installation complete.
pause
