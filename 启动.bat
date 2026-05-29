@echo off
chcp 65001 >nul
title 水质监测溯源助手
cd /d "%~dp0"

echo ============================================
echo   水质监测溯源助手 — 一键启动
echo ============================================
echo.

:: Step 1: find Python
set PYCMD=
py --version >nul 2>&1
if not errorlevel 1 set PYCMD=py
if "%PYCMD%"=="" python --version >nul 2>&1
if "%PYCMD%"=="" if not errorlevel 1 set PYCMD=python
if "%PYCMD%"=="" python3 --version >nul 2>&1
if "%PYCMD%"=="" if not errorlevel 1 set PYCMD=python3

if "%PYCMD%"=="" (
    echo [错误] 未找到 Python，请先安装 Python 3.10 或以上版本
    echo        下载地址: https://www.python.org/downloads/
    echo        安装时请务必勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [检测] Python 命令: %PYCMD%
%PYCMD% --version
echo.

:: Step 2: install dependencies
echo [检查] 正在检查依赖包...
%PYCMD% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [安装] 首次运行，正在安装依赖包（约 1-2 分钟）...
    echo.
    %PYCMD% -m pip install --user -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [错误] 依赖包安装失败，请检查网络连接后重试
        pause
        exit /b 1
    )
    echo.
    echo [完成] 依赖包安装成功
) else (
    echo [跳过] 依赖包已安装
)

:: Step 2.5: install MCP sub-package
echo [检查] MCP 子包...
%PYCMD% -c "import water_quality_mcp" >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 MCP 子包...
    %PYCMD% -m pip install --user -e water_quality_mcp >nul 2>&1
    if errorlevel 1 (
        echo [警告] MCP 子包安装失败，Streamlit 主界面仍可正常使用
    ) else (
        echo [完成] MCP 子包安装成功
    )
) else (
    echo [跳过] MCP 子包已安装
)
echo.

:: Step 3: launch — try port 8501 first, fall back to 8502, 8503
call :try_port 8501 || call :try_port 8502 || call :try_port 8503 || (
    echo [错误] 端口 8501-8503 均被占用，请手动释放后重试
    pause
    exit /b 1
)
goto :eof

:try_port
set PORT=%1
netstat -ano | findstr ":%PORT% " >nul 2>&1
if errorlevel 1 (
    echo [启动] 正在启动，浏览器将自动打开...
    echo        地址: http://localhost:%PORT%
    echo        按 Ctrl+C 可停止运行
    echo.
    %PYCMD% -m streamlit run app.py --server.port %PORT%
    exit /b 0
)
echo [提示] 端口 %PORT% 已被占用，尝试下一个...
exit /b 1
