# -*- encoding: utf-8 -*-
import os
import time
import websockets
import ssl
import asyncio
import json
import traceback
import argparse
import logging
import datetime # Added
import socket # Added
import uuid # Added
from pathlib import Path # Added

# --- Database Imports ---
import pymysql # Added
from pymysql.cursors import DictCursor # Added

# --- API Specific Imports ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Request # Added Request
from fastapi.responses import JSONResponse
import aiofiles # For async file writing
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Configuration ---
# !! IMPORTANT: Replace with your actual DB credentials or load from config/env !!
DB_CONFIG = {
    'host': '***********.sql.tencentcdb.com',
    'port': *****,
    'user': '******',
    'password': '******',
    'database': 'your_database_name', # <--- MUST BE REPLACED
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

# Directory to save uploaded audio files for potential review
# !! IMPORTANT: Ensure this directory exists and the server process has write permissions !!
AUDIO_SAVE_DIR = Path("./saved_audio")
AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FunASR HTTP API with DB Logging",
    description="An API wrapper for the FunASR WebSocket streaming client that logs requests and saves audio.",
)

# --- Helper Functions ---

def chinese_json_dumps(obj):
    """Ensure JSON strings with Chinese characters are not escaped."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)

def get_server_ip():
    """Gets the primary IP address of the server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)) # Connect to a known external server
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        log.warning(f"Could not determine server IP automatically: {e}. Using 127.0.0.1.")
        return "127.0.0.1"

SERVER_IP = get_server_ip() # Get server IP once on startup

async def save_audio_file(audio_bytes: bytes, filename: str, req_seqno: str) -> Optional[Path]:
    """Saves audio bytes to a uniquely named file in AUDIO_SAVE_DIR."""
    try:
        # Create a unique filename to avoid collisions
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_original_filename = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in filename)
        save_filename = f"{timestamp}_{req_seqno}_{safe_original_filename}"
        save_path = AUDIO_SAVE_DIR / save_filename

        async with aiofiles.open(save_path, "wb") as f:
            await f.write(audio_bytes)
        log.info(f"Saved uploaded audio to: {save_path}")
        return save_path
    except Exception as e:
        log.error(f"Failed to save audio file {filename} for REQ_SEQNO {req_seqno}: {e}")
        log.error(traceback.format_exc())
        return None

# Synchronous function to be run in a thread pool for DB operations
def save_log_to_db_sync(log_data: dict):
    """Connects to the DB and inserts a single log entry."""
    connection = None
    if DB_CONFIG['database'] == 'your_database_name':
         log.error("Database name 'your_database_name' is not configured. Skipping DB log.")
         return # Don't attempt to connect if DB name is default

    try:
        log.debug(f"Connecting to DB to log REQ_SEQNO: {log_data.get('REQ_SEQNO')}")
        connection = pymysql.connect(**DB_CONFIG)
        sql = """
        INSERT INTO save_asr_data (
            TRAN_DATE, TRAN_TIMESTAMP, REQ_SEQNO, REQ_CHANNELID, REQ_BANK_CODE,
            REQ_USERID, REQ_SERVICEID, REQ_MESSAGE, REQ_DOCUMENT, RES_ERROR_CODE,
            RES_ERROR_MESSAGE, RES_DOCUMENT, RES_MESSAGE, SERVER_IP, TOTAL_TIME_USED,
            ASR_FIND_WAV_PATH_STATUS, ASR_VOICERESULT, ASR_WAV_PATH
        ) VALUES (
            %(TRAN_DATE)s, %(TRAN_TIMESTAMP)s, %(REQ_SEQNO)s, %(REQ_CHANNELID)s, %(REQ_BANK_CODE)s,
            %(REQ_USERID)s, %(REQ_SERVICEID)s, %(REQ_MESSAGE)s, %(REQ_DOCUMENT)s, %(RES_ERROR_CODE)s,
            %(RES_ERROR_MESSAGE)s, %(RES_DOCUMENT)s, %(RES_MESSAGE)s, %(SERVER_IP)s, %(TOTAL_TIME_USED)s,
            %(ASR_FIND_WAV_PATH_STATUS)s, %(ASR_VOICERESULT)s, %(ASR_WAV_PATH)s
        )
        """
        with connection.cursor() as cursor:
            # Ensure JSON fields are properly formatted strings or None
            log_data['REQ_MESSAGE'] = chinese_json_dumps(log_data.get('REQ_MESSAGE'))
            log_data['REQ_DOCUMENT'] = chinese_json_dumps(log_data.get('REQ_DOCUMENT'))
            log_data['RES_DOCUMENT'] = chinese_json_dumps(log_data.get('RES_DOCUMENT'))
            log_data['RES_MESSAGE'] = chinese_json_dumps(log_data.get('RES_MESSAGE'))
            log_data['ASR_VOICERESULT'] = chinese_json_dumps(log_data.get('ASR_VOICERESULT'))
            # Convert Path object to string for DB
            if isinstance(log_data.get('ASR_WAV_PATH'), Path):
                 log_data['ASR_WAV_PATH'] = str(log_data['ASR_WAV_PATH'])

            cursor.execute(sql, log_data)
        connection.commit()
        log.info(f"Successfully logged request to DB, REQ_SEQNO: {log_data.get('REQ_SEQNO')}")

    except pymysql.Error as db_err:
        log.error(f"Database error while logging REQ_SEQNO {log_data.get('REQ_SEQNO')}: {db_err}")
        log.error(traceback.format_exc())
        if connection:
            connection.rollback()
    except Exception as e:
        log.error(f"Unexpected error during DB logging for REQ_SEQNO {log_data.get('REQ_SEQNO')}: {e}")
        log.error(traceback.format_exc())
        if connection:
            connection.rollback()
    finally:
        if connection:
            connection.close()
            log.debug(f"DB connection closed for REQ_SEQNO: {log_data.get('REQ_SEQNO')}")

# --- WebSocket Transcription Logic (Mostly unchanged, added logging) ---
async def transcribe_audio_ws(
    host: str,
    port: int,
    chunk_size: List[int],
    chunk_interval: int,
    hotword: str,
    audio_bytes: bytes,
    audio_fs: int,
    wav_name: str,
    ssl_enabled: bool,
    use_itn: bool,
    mode: str,
    req_seqno: str, # Added for logging context
):
    """
    Connects to FunASR WebSocket, sends audio, and returns the transcription result.
    """
    result = None
    final_message_received = asyncio.Event()

    fst_dict = {}
    hotword_msg = ""
    if hotword and hotword.strip(): # Added check for non-empty hotword
        try:
            lines = hotword.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        fst_dict[" ".join(parts[:-1])] = int(parts[-1])
                    except ValueError:
                        log.warning(f"[{req_seqno}] Skipping invalid hotword line: {line}")
                elif line.strip():
                     log.warning(f"[{req_seqno}] Skipping invalid hotword line format: {line}")
            if fst_dict:
                hotword_msg = json.dumps(fst_dict) # No chinese_json_dumps needed here, FunASR expects standard JSON
                log.info(f"[{req_seqno}] Using hotwords: {hotword_msg}")
            else:
                log.warning(f"[{req_seqno}] No valid hotwords parsed from input: {hotword}")
        except Exception as e:
            log.error(f"[{req_seqno}] Error processing hotwords: {e}")
            hotword_msg = ""

    if ssl_enabled:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        uri = f"wss://{host}:{port}"
    else:
        uri = f"ws://{host}:{port}"
        ssl_context = None

    log.info(f"[{req_seqno}] Connecting to {uri} with mode: {mode}")

    try:
        async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context, max_size=None) as websocket:

            async def send_audio():
                try:
                    config_message = json.dumps({
                        "mode": mode,
                        "chunk_size": chunk_size,
                        "chunk_interval": chunk_interval,
                        "audio_fs": audio_fs,
                        "wav_name": wav_name,
                        "wav_format": "pcm",
                        "is_speaking": True,
                        "hotwords": hotword_msg,
                        "itn": use_itn,
                    })
                    log.debug(f"[{req_seqno}] Sending config: {config_message}")
                    await websocket.send(config_message)

                    if audio_fs <= 0 or chunk_interval <= 0:
                       log.error(f"[{req_seqno}] Invalid audio_fs ({audio_fs}) or chunk_interval ({chunk_interval})")
                       raise ValueError("Invalid audio parameters for stride calculation")

                    stride = int(60 * chunk_size[1] / chunk_interval / 1000 * audio_fs * 2)
                    if stride <= 0:
                         log.warning(f"[{req_seqno}] Calculated stride is {stride}. Sending audio in one chunk.")
                         stride = len(audio_bytes)

                    if not audio_bytes:
                         log.warning(f"[{req_seqno}] Audio data is empty.")
                         chunk_num = 0
                    elif stride <= 0:
                        log.warning(f"[{req_seqno}] Stride zero or negative. Sending as single chunk.")
                        chunk_num = 1
                        stride = len(audio_bytes)
                    else:
                        chunk_num = (len(audio_bytes) - 1) // stride + 1

                    log.info(f"[{req_seqno}] Audio size: {len(audio_bytes)} bytes, Stride: {stride}, Chunks: {chunk_num}")

                    for i in range(chunk_num):
                        beg = i * stride
                        end = min(beg + stride, len(audio_bytes))
                        data = audio_bytes[beg:end]
                        if not data:
                            log.warning(f"[{req_seqno}] Skipping empty chunk {i+1}/{chunk_num}")
                            continue
                        log.debug(f"[{req_seqno}] Sending chunk {i+1}/{chunk_num}, size: {len(data)}")
                        await websocket.send(data)
                        sleep_duration = 0.001 if mode == "offline" else (60 * chunk_size[1] / chunk_interval / 1000)
                        await asyncio.sleep(max(0.001, sleep_duration * 0.9))

                    final_chunk_message = json.dumps({"is_speaking": False})
                    log.debug(f"[{req_seqno}] Sending final chunk marker: {final_chunk_message}")
                    await websocket.send(final_chunk_message)
                    log.info(f"[{req_seqno}] Finished sending audio data.")

                except Exception as e:
                    log.error(f"[{req_seqno}] Error during send_audio task: {e}")
                    log.error(traceback.format_exc())
                    if not final_message_received.is_set(): final_message_received.set()

            async def receive_messages():
                nonlocal result
                try:
                    while True:
                        meg_str = await websocket.recv()
                        log.debug(f"[{req_seqno}] Received message: {meg_str}")
                        try:
                            meg = json.loads(meg_str)
                        except json.JSONDecodeError:
                            log.error(f"[{req_seqno}] Failed to decode JSON from server: {meg_str}")
                            continue

                        is_final = meg.get("is_final", False)

                        if mode == "offline":
                            result = meg
                            final_message_received.set()
                            log.info(f"[{req_seqno}] Offline result received: {result.get('text')}")
                            break
                        elif mode == "2pass":
                            if is_final:
                                result = meg
                                final_message_received.set()
                                log.info(f"[{req_seqno}] 2Pass final result received: {result.get('text')}")
                                break
                            else:
                                log.debug(f"[{req_seqno}] Ignoring intermediate 2pass message.")
                        elif mode == "online":
                             result = meg
                             if is_final:
                                final_message_received.set()
                                log.info(f"[{req_seqno}] Online final result received: {result.get('text')}")
                                break
                             else:
                                 log.debug(f"[{req_seqno}] Received intermediate online result.")
                        else:
                             log.warning(f"[{req_seqno}] Received message for unknown mode config: {meg.get('mode')}")
                             result = meg
                             if is_final:
                                 final_message_received.set()
                                 break

                except websockets.exceptions.ConnectionClosedOK:
                    log.info(f"[{req_seqno}] WebSocket connection closed gracefully by server.")
                    if not result and not final_message_received.is_set(): # If closed before any result
                        final_message_received.set() # Allow main wait to finish
                except websockets.exceptions.ConnectionClosedError as e:
                    log.error(f"[{req_seqno}] WebSocket connection closed with error: {e}")
                    if not final_message_received.is_set(): final_message_received.set()
                except Exception as e:
                    log.error(f"[{req_seqno}] Error receiving message: {e}")
                    log.error(traceback.format_exc())
                    if not final_message_received.is_set(): final_message_received.set()
                finally:
                    if not final_message_received.is_set():
                        log.warning(f"[{req_seqno}] Receive loop finished unexpectedly.")
                        final_message_received.set()

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_messages())

            timeout_seconds = 180
            try:
                log.debug(f"[{req_seqno}] Waiting for final message signal (timeout: {timeout_seconds}s)")
                await asyncio.wait_for(final_message_received.wait(), timeout=timeout_seconds)
                log.info(f"[{req_seqno}] Final message signal received or receive task ended.")
            except asyncio.TimeoutError:
                 log.error(f"[{req_seqno}] Timeout waiting for transcription result after {timeout_seconds}s.")
                 if not send_task.done(): send_task.cancel()
                 if not receive_task.done(): receive_task.cancel()
                 await asyncio.gather(send_task, receive_task, return_exceptions=True)
                 raise HTTPException(status_code=504, detail=f"Timeout waiting for FunASR result (REQ_SEQNO: {req_seqno})")

            await asyncio.gather(send_task, receive_task, return_exceptions=True)

            if send_task.done() and send_task.exception():
                 log.error(f"[{req_seqno}] Send task finished with exception: {send_task.exception()}")
            if receive_task.done() and receive_task.exception():
                 log.error(f"[{req_seqno}] Receive task finished with exception: {receive_task.exception()}")

    except websockets.exceptions.InvalidURI:
        log.error(f"[{req_seqno}] Invalid WebSocket URI: {uri}")
        raise HTTPException(status_code=400, detail=f"Invalid WebSocket URI format for REQ_SEQNO: {req_seqno}")
    except websockets.exceptions.WebSocketException as e:
        log.error(f"[{req_seqno}] WebSocket connection failed: {e}")
        raise HTTPException(status_code=502, detail=f"Could not connect to FunASR WebSocket (REQ_SEQNO: {req_seqno}): {e}")
    except Exception as e:
        log.error(f"[{req_seqno}] An unexpected error occurred during WebSocket interaction: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error during WebSocket interaction (REQ_SEQNO: {req_seqno}): {e}")

    if result:
        return result
    else:
        log.error(f"[{req_seqno}] No valid result received from FunASR service.")
        raise HTTPException(status_code=502, detail=f"No result received from FunASR (REQ_SEQNO: {req_seqno}). Check logs.")

# --- API Endpoint Definition ---
@app.post("/transcribe", response_class=JSONResponse)
async def http_transcribe(
    request: Request, # Inject request object to get client IP if needed, though using server IP now
    # --- Request Metadata (Added for DB logging) ---
    req_seqno: Optional[str] = Query(None, description="Unique request sequence number (optional, will be generated if empty)."),
    req_channelid: Optional[str] = Query("API", description="Request channel ID."),
    req_bank_code: Optional[str] = Query(None, description="Bank code associated with the request."),
    req_userid: Optional[str] = Query(None, description="User ID associated with the request."),
    req_serviceid: Optional[str] = Query("ASR", description="Service ID."),
    # --- FunASR Parameters ---
    host: str = Query("127.0.0.1", description="FunASR WebSocket server host IP."),
    port: int = Query(10098, description="FunASR WebSocket server port."),
    mode: str = Query("offline", description="Recognition mode: offline, online, 2pass.", pattern="^(offline|online|2pass)$"),
    chunk_size_str: str = Query("5,10,5", alias="chunk_size", description="Chunk size tuple (comma-separated): 'before,middle,after' e.g., 5,10,5."),
    chunk_interval: int = Query(10, description="Chunk processing interval in ms."),
    hotword: Optional[str] = Form(None, description="Hotwords string, one per line (e.g., '阿里巴巴 20\\n腾讯 15'). URL Encode if needed."),
    audio_fs: int = Query(16000, description="Audio sample rate in Hz."),
    ssl_enabled: bool = Query(False, alias="ssl", description="Use wss:// secure connection."),
    use_itn: bool = Query(True, description="Apply Inverse Text Normalization."),
    # --- Audio Input ---
    audio_file: UploadFile = File(..., description="Audio file to transcribe (e.g., wav, pcm)."),
):
    """
    Receives an audio file and parameters, interacts with the FunASR WebSocket service,
    logs the request/response to the database, saves the audio file,
    and returns the transcription result as JSON.
    """
    start_time = time.time()
    tran_date = datetime.datetime.now().strftime('%Y%m%d')
    tran_timestamp = datetime.datetime.now().strftime('%H%M%S%f')[:9]

    # Generate req_seqno if not provided
    if not req_seqno:
        req_seqno = f"API_{uuid.uuid4()}" # Generate a unique ID
    log.info(f"[{req_seqno}] Received request. File: {audio_file.filename}, Mode: {mode}")

    # --- Initialize log data ---
    log_data = {
        'TRAN_DATE': tran_date,
        'TRAN_TIMESTAMP': tran_timestamp,
        'REQ_SEQNO': req_seqno,
        'REQ_CHANNELID': req_channelid,
        'REQ_BANK_CODE': req_bank_code,
        'REQ_USERID': req_userid,
        'REQ_SERVICEID': req_serviceid,
        # Capture request parameters (excluding file)
        'REQ_MESSAGE': {
            "host": host, "port": port, "mode": mode, "chunk_size": chunk_size_str,
            "chunk_interval": chunk_interval, "hotword_provided": bool(hotword),
            "audio_fs": audio_fs, "ssl": ssl_enabled, "use_itn": use_itn,
            "original_filename": audio_file.filename
        },
        'REQ_DOCUMENT': None, # Not used in this endpoint
        'RES_ERROR_CODE': None, # Will be set based on outcome
        'RES_ERROR_MESSAGE': None, # Will be set based on outcome
        'RES_DOCUMENT': None, # Not used
        'RES_MESSAGE': None, # Will contain ASR result or error details
        'SERVER_IP': SERVER_IP,
        'TOTAL_TIME_USED': None, # Will be calculated
        'ASR_FIND_WAV_PATH_STATUS': 'N', # Default to No
        'ASR_VOICERESULT': None, # Will contain ASR result
        'ASR_WAV_PATH': None # Will be set after saving
    }

    saved_audio_path = None
    audio_bytes = None
    transcription_result = None
    http_exception = None

    try:
        # --- Parameter Parsing and Validation ---
        try:
            chunk_size = [int(x.strip()) for x in chunk_size_str.split(",")]
            if len(chunk_size) != 3:
                raise ValueError("chunk_size must have 3 comma-separated integers.")
        except ValueError as e:
            log.error(f"[{req_seqno}] Invalid chunk_size format: {chunk_size_str}. Error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid chunk_size format: '{chunk_size_str}'. Must be 'int,int,int'.")

        # --- Read and Save Audio File ---
        try:
            audio_bytes = await audio_file.read()
            wav_name = audio_file.filename if audio_file.filename else f"audio_{req_seqno}.unknown"
            log.info(f"[{req_seqno}] Read {len(audio_bytes)} bytes from {wav_name}")
            if not audio_bytes:
                 raise HTTPException(status_code=400, detail=f"Uploaded audio file is empty (REQ_SEQNO: {req_seqno}).")

            # Save the audio asynchronously
            saved_audio_path = await save_audio_file(audio_bytes, wav_name, req_seqno)
            if saved_audio_path:
                log_data['ASR_WAV_PATH'] = saved_audio_path # Store Path object for now
                log_data['ASR_FIND_WAV_PATH_STATUS'] = 'Y'

        except HTTPException as he: # Re-raise validation errors
            raise he
        except Exception as e:
            log.error(f"[{req_seqno}] Failed to read or save uploaded file: {e}")
            log.error(traceback.format_exc())
            # Don't raise immediately, log this attempt with error, but allow DB logging
            http_exception = HTTPException(status_code=400, detail=f"Could not read/save audio file (REQ_SEQNO: {req_seqno}): {e}")
            # Set error details for logging
            log_data['RES_ERROR_CODE'] = str(http_exception.status_code) # Use HTTP status code
            log_data['RES_ERROR_MESSAGE'] = http_exception.detail[:30] # Truncate message if too long
            # Still need to ensure audio_file is closed if opened
        finally:
            if audio_file:
                 await audio_file.close() # Ensure file handle is closed

        # If reading/saving failed, stop processing and log the error
        if http_exception:
             raise http_exception # Raise the caught exception

        # --- Call the WebSocket Interaction Logic ---
        transcription_result = await transcribe_audio_ws(
            host=host,
            port=port,
            chunk_size=chunk_size,
            chunk_interval=chunk_interval,
            hotword=hotword if hotword else "",
            audio_bytes=audio_bytes,
            audio_fs=audio_fs,
            wav_name=wav_name,
            ssl_enabled=ssl_enabled,
            use_itn=use_itn,
            mode=mode,
            req_seqno=req_seqno, # Pass seqno for logging context
        )

        # --- Success Case ---
        log_data['RES_ERROR_CODE'] = '0000'
        log_data['RES_ERROR_MESSAGE'] = '成功'
        log_data['RES_MESSAGE'] = transcription_result # Store the full result
        log_data['ASR_VOICERESULT'] = transcription_result # Also store in ASR specific field

        end_time = time.time()
        total_time = f"{end_time - start_time:.3f}"
        log_data['TOTAL_TIME_USED'] = total_time
        log.info(f"[{req_seqno}] Successfully transcribed {wav_name} in {total_time} seconds")

        # Return the ASR result
        return JSONResponse(content=transcription_result)

    except HTTPException as e:
        # Handle exceptions raised from validation or transcribe_audio_ws
        http_exception = e # Store the exception to re-raise later
        log.warning(f"[{req_seqno}] Transcription failed for {audio_file.filename if audio_file else 'N/A'} with HTTPException: {e.status_code} - {e.detail}")
        log_data['RES_ERROR_CODE'] = str(e.status_code)
        log_data['RES_ERROR_MESSAGE'] = e.detail[:30] # Truncate if needed
        # Store error detail in RES_MESSAGE as well for consistency
        log_data['RES_MESSAGE'] = {"error": e.detail, "status_code": e.status_code}

    except Exception as e:
        # Catch any other unexpected errors
        http_exception = HTTPException(status_code=500, detail=f"Internal server error during transcription (REQ_SEQNO: {req_seqno})")
        log.error(f"[{req_seqno}] Unexpected error during transcription process for {audio_file.filename if audio_file else 'N/A'}: {e}")
        log.error(traceback.format_exc())
        log_data['RES_ERROR_CODE'] = '9999' # Generic internal error code
        log_data['RES_ERROR_MESSAGE'] = '内部服务器错误' # Generic message
        log_data['RES_MESSAGE'] = {"error": str(e), "status_code": 500}

    finally:
        # --- Log to Database (always attempt, regardless of success/failure) ---
        if log_data['TOTAL_TIME_USED'] is None: # Calculate time if not already set (e.g., due to error)
            end_time = time.time()
            log_data['TOTAL_TIME_USED'] = f"{end_time - start_time:.3f}"

        # Use asyncio.to_thread to run the blocking DB operation without blocking the event loop
        log_task = asyncio.create_task(asyncio.to_thread(save_log_to_db_sync, log_data.copy()))
        # We don't necessarily need to await log_task here unless we want to guarantee logging before returning an error
        # For now, let it run in the background. Add await if strict logging order is needed.
        # await log_task # Uncomment this if you need to ensure logging completes before response

        # If an exception occurred, re-raise it now after attempting to log
        if http_exception:
            raise http_exception


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FunASR HTTP API Server with DB Logging")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="Host for the API server")
    parser.add_argument("--api_port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers (for production)")
    parser.add_argument("--audio_dir", type=str, default="./saved_audio", help="Directory to save uploaded audio files")
    # Add DB config arguments if you want to override DB_CONFIG via CLI
    # parser.add_argument("--db_host", type=str, default=DB_CONFIG['host'])
    # ... etc ...
    api_args = parser.parse_args()

    # Update global config based on args if provided
    AUDIO_SAVE_DIR = Path(api_args.audio_dir)
    AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Audio files will be saved to: {AUDIO_SAVE_DIR.resolve()}")

    # Update DB_CONFIG if CLI args are added and provided
    # if api_args.db_host: DB_CONFIG['host'] = api_args.db_host
    # ... etc ...

    if DB_CONFIG['database'] == 'your_database_name':
        log.warning("=" * 60)
        log.warning("!!! IMPORTANT: Database name in DB_CONFIG is set to the default 'your_database_name'.")
        log.warning("!!! Please edit the script and set the correct database name before running.")
        log.warning("!!! Database logging will be skipped until configured.")
        log.warning("=" * 60)


    print(f"Starting FunASR API Server on http://{api_args.api_host}:{api_args.api_port}")
    print(f"Using {api_args.workers} worker(s).")
    print(f"Saving audio files to: {AUDIO_SAVE_DIR.resolve()}")
    print(f"Logging to database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']} (User: {DB_CONFIG['user']})")


    # Use the name of the current file (assuming it's saved as main.py)
    # Format is "module_name:app_instance_name"
    uvicorn.run(
        "main:app",
        host=api_args.api_host,
        port=api_args.api_port,
        workers=api_args.workers,
        # reload=False # Disable reload for production/multi-worker setups
    )


"""
测试单声道识别：
curl -X POST \
  "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&chunk_size=5,10,5&chunk_interval=10&ssl=false&use_itn=true&audio_fs=16000&req_seqno=TEST001&req_channelid=CURL_BASIC" \
  -H "accept: application/json" \
  -F "audio_file=@../audio/asr_example.wav"  # <--- IMPORTANT: Adjust this path to your audio file!
"""