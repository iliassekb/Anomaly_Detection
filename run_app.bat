@echo off
cd /d "%~dp0"
pfe_env\Scripts\python.exe -m streamlit run app.py %*
