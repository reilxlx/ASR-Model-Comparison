#!/bin/bash

# 配置参数
HOST="0.0.0.0"
PORT="8000"
WORKERS=10
LOG_CONFIG="log_config.yaml"
LOG_FILE="server_stdout.log"
PID_FILE="uvicorn.pid"

# 启动命令
echo "Starting Uvicorn server..."
nohup uvicorn main:app \
  --host $HOST \
  --port $PORT \
  --workers $WORKERS \
  --log-config $LOG_CONFIG > $LOG_FILE 2>&1 &

# 保存进程号
echo $! > $PID_FILE
echo "Uvicorn server started with PID $(cat $PID_FILE)"
echo "Logs are being written to $LOG_FILE"