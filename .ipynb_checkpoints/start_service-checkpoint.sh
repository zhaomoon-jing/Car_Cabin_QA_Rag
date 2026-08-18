#!/bin/bash
cd $(dirname $0)
export PYTHONUNBUFFERED=1
nohup python3 car_api.py > service.log 2>&1 &
echo $! > service.pid
echo "car‑cockpit‑rag 服务已后台启动，pid写入service.pid"