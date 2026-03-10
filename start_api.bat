@echo off
    chcp 65001 >nul
    cd /d "D:\projects\langgraph-agent"
    call "D:\projects\langgraph-agent\venv\Scripts\activate.bat"
    set OLLAMA_BASE_URL=http://localhost:11435
    echo ========================================
    echo  LangGraph Agent API 服务
    echo ========================================
    echo 启动中...
    python -m src.api.main
    pause
    