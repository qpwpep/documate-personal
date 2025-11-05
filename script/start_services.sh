#!/bin/bash

# 현재 스크립트가 위치한 디렉토리로 이동합니다.
BASE_DIR=$(dirname "$0")
cd "$BASE_DIR"
cd ".."

echo "FastAPI 서비스를 백그라운드에서 시작합니다 (포트 8000)..."

# FastAPI 서비스 실행 (uvicorn new_src.web.main:app)
# nohup: 터미널 종료 후에도 프로세스 유지
# > fastapi.log 2>&1 &: 모든 출력 및 에러를 fastapi.log에 기록하고 백그라운드(&) 실행
uv run nohup uvicorn new_src.web.main:app --reload --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
echo "FastAPI PID: $! (로그 파일: fastapi.log)"


echo "Streamlit 앱을 백그라운드에서 시작합니다 (포트 8501)..."

# Streamlit 앱 실행 (streamlit run new_src/web/streamlit_app.py)
# nohup과 로그 파일 기록 방식은 FastAPI와 동일합니다.
uv run nohup streamlit run new_src/web/streamlit_app.py --server.port 8501 > streamlit.log 2>&1 &
echo "Streamlit PID: $! (로그 파일: streamlit.log)"


echo "============================================================"
echo "두 웹 서비스가 백그라운드에서 성공적으로 실행되었습니다."
echo "서비스 종료는 '. script/stop_services.sh' 또는"
echo "'uv run python -m new_src.main --mode stopweb' 를 실행해주세요." 
echo "============================================================"