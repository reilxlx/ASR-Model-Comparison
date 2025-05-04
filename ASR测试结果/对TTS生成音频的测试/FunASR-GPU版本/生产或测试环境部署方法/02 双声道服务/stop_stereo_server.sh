#!/bin/bash

# --- Configuration (Must match start_stereo_server.sh) ---
PID_FILE="stereo_uvicorn.pid"

echo "Attempting to stop Stereo ASR API server..."

# --- Check if PID file exists ---
if [ ! -f "$PID_FILE" ]; then
    echo "Error: PID file $PID_FILE not found. Server may not be running or was not started with start_stereo_server.sh."
    exit 1
fi

# --- Read PID ---
PID=$(cat "$PID_FILE")

if [ -z "$PID" ]; then
    echo "Error: PID file $PID_FILE is empty."
    rm -f "$PID_FILE" # Clean up empty file
    exit 1
fi

# --- Check if process is running ---
if ps -p $PID > /dev/null; then
    echo "Found running server with PID $PID. Sending SIGTERM..."
    kill $PID # Send TERM signal for graceful shutdown

    # --- Wait for graceful shutdown ---
    WAIT_SECONDS=10
    echo -n "Waiting up to $WAIT_SECONDS seconds for graceful shutdown."
    for ((i=0; i<WAIT_SECONDS; i++)); do
        if ! ps -p $PID > /dev/null; then
            echo "" # Newline after dots
            echo "Server with PID $PID stopped gracefully."
            rm -f "$PID_FILE" # Remove PID file only on success
            exit 0
        fi
        echo -n "."
        sleep 1
    done
    echo "" # Newline after dots

    # --- Force kill if still running ---
    if ps -p $PID > /dev/null; then
        echo "Server did not stop gracefully after $WAIT_SECONDS seconds. Sending SIGKILL (force kill)..."
        kill -9 $PID
        sleep 1 # Give OS a moment

        if ! ps -p $PID > /dev/null; then
            echo "Server with PID $PID force killed."
            rm -f "$PID_FILE" # Remove PID file
            exit 0
        else
            echo "Error: Failed to kill process $PID even with SIGKILL. Manual intervention required."
            exit 1
        fi
    fi
else
    echo "Warning: Process with PID $PID specified in $PID_FILE is not running."
    echo "Cleaning up PID file: $PID_FILE"
    rm -f "$PID_FILE"
    exit 0 # Exit successfully as the desired state (stopped) is achieved
fi