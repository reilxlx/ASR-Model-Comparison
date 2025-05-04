#!/bin/bash

# --- Configuration for Stereo ASR API ---
HOST="0.0.0.0"
# Ensure this port is DIFFERENT from your mono API server (e.g., main.py's server)
PORT="8001"
# Adjust workers based on your server's CPU cores (e.g., 2 * cores + 1)
WORKERS=4
# Optional: Specify a log config YAML if needed, otherwise Uvicorn uses defaults
LOG_CONFIG="stereo_log_config.yaml"
LOG_FILE="stereo_server_stdout.log"
PID_FILE="stereo_uvicorn.pid"
APP_MODULE="stereo_asr_api_v2:app" # Correctly points to your stereo API file and FastAPI app object

# --- Check if PID file exists and process is running ---
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "Stereo ASR API server already running with PID $PID."
        exit 1
    else
        echo "Warning: PID file $PID_FILE found, but process $PID is not running. Overwriting PID file."
        rm -f "$PID_FILE"
    fi
fi

# --- Create log config if it doesn't exist (basic example) ---
# You might want a more sophisticated YAML for production
if [ ! -f "$LOG_CONFIG" ]; then
  echo "Creating basic log config file: $LOG_CONFIG"
  cat << EOF > $LOG_CONFIG
version: 1
disable_existing_loggers: false
formatters:
  default:
    fmt: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
handlers:
  console:
    class: logging.StreamHandler
    formatter: default
    stream: ext://sys.stderr
loggers:
  uvicorn.error:
    level: INFO
    handlers: [console]
    propagate: no
  uvicorn.access:
    level: INFO
    handlers: [console]
    propagate: no
  stereo_asr_api: # Match the logger name used in your python script if specific
    level: INFO
    handlers: [console]
    propagate: yes
EOF
fi


# --- Start Command ---
echo "Starting Stereo ASR API server ($APP_MODULE)..."
# Using --log-config is generally preferred over redirecting stdout/stderr for structured logging
nohup uvicorn $APP_MODULE \
  --host $HOST \
  --port $PORT \
  --workers $WORKERS \
  --log-config $LOG_CONFIG > $LOG_FILE 2>&1 &

# --- Save PID and Provide Feedback ---
# Ensure the process started successfully before writing PID
sleep 1 # Give uvicorn a moment to potentially fail early
SERVER_PID=$!

if ps -p $SERVER_PID > /dev/null; then
    echo $SERVER_PID > $PID_FILE
    echo "Stereo ASR API server started successfully with PID $SERVER_PID."
    echo "Logs are configured via $LOG_CONFIG (stdout/stderr redirected to $LOG_FILE as fallback)."
else
    echo "Error: Failed to start Stereo ASR API server. Check $LOG_FILE for details."
    exit 1
fi

exit 0
