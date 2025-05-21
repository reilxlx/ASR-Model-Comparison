#!/bin/bash

PID_FILE="uvicorn.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    echo "Stopping Uvicorn server with PID $PID"
    kill $PID
    rm $PID_FILE
else
    echo "PID file not found. Is the server running?"
fi