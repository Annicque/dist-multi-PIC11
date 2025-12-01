@echo off
REM Lancer distillation_multicomposants.py avec l'environnement virtuel
cd /d "%~dp0"
call env\Scripts\activate.bat
python distillation_multicomposants.py %*
pause
