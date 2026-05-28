@echo off
chcp 65001 >nul
title 水质监测溯源助手
cd /d "%~dp0"

:: Check if .venv exists, if not create it
if not exist ".venv\Scripts\python.exe" (
    echo [提示] 未检测到虚拟环境，正在创建...
    python -m venv .venv
    echo.
)

:: Activate venv and check dependencies
call ".venv\Scripts\activate.bat"

:: Check if streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖包，请稍候...
    pip install -r requirements.txt
    echo.
)

echo [启动] 水质监测溯源助手...
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 可停止运行
echo.

streamlit run app.py --server.port 8501
pause
