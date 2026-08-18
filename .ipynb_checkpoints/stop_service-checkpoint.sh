#!/bin/bash
PID=$(cat service.pid)
kill $PID
rm -f service.pid
echo "服务停止"