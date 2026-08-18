#!/bin/bash
# 车载离线RAG服务启动脚本
BASE_DIR=$(cd $(dirname $0);pwd)
VENV_PATH="${BASE_DIR}/venv/bin/python"
cd ${BASE_DIR}

echo "启动FastAPI RAG服务 端口8000"
${VENV_PATH} -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1