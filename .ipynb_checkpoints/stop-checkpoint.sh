#!/bin/bash
pid=$(ps aux | grep "uvicorn main:app" | grep -v grep | awk '{print $2}')
if [ -n "$pid" ];then
    kill -15 $pid
    echo "服务进程 $pid 已停止"
else
    echo "未检测到运行中的服务"
fi