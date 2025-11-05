#!/bin/bash

echo "실행 중인 웹 서비스 종료를 시도합니다..."

# --- FastAPI 프로세스 종료 시도 ---
# 명령어에 "uvicorn src.web.main:app" 패턴이 포함된 프로세스를 pkill로 종료합니다.
pkill -f "uvicorn src.web.main:app"
if [ $? -eq 0 ]; then
    echo "FastAPI (uvicorn) 관련 프로세스가 종료되었습니다."
else
    echo "FastAPI 실행 중인 프로세스를 찾지 못했거나 이미 종료되었습니다."
fi

# --- Streamlit 프로세스 종료 시도 ---
# 명령어에 "streamlit run src/web/streamlit_app.py" 패턴이 포함된 프로세스를 pkill로 종료합니다.
pkill -f "streamlit run src/web/streamlit_app.py"
if [ $? -eq 0 ]; then
    echo "Streamlit 관련 프로세스가 종료되었습니다."
else
    echo "Streamlit 실행 중인 프로세스를 찾지 못했거나 이미 종료되었습니다."
fi

echo "============================================================"
echo "모든 서비스 종료 시도 완료."
echo "============================================================"