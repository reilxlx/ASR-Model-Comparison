# -*- encoding: utf-8 -*-
import os
import time
import websockets
import ssl
import asyncio
import json
import traceback
import argparse
# from multiprocessing import Process # Removed as not strictly needed for API model
import logging
# from queue import Queue # Removed as not strictly needed for API model

# --- API Specific Imports ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import JSONResponse
import aiofiles # For async file reading if needed, though UploadFile.read() is often sufficient
from typing import List, Optional

logging.basicConfig(level=logging.INFO) # Changed to INFO for better debugging
log = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
# This 'app' object is what uvicorn needs to find
app = FastAPI(
    title="FunASR HTTP API",
    description="An API wrapper for the FunASR WebSocket streaming client.",
)

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
):
    """
    Connects to FunASR WebSocket, sends audio, and returns the transcription result.
    """
    result = None
    final_message_received = asyncio.Event() # Event to signal completion

    # Prepare hotwords
    fst_dict = {}
    hotword_msg = ""
    if hotword.strip():
        try:
            # Assuming hotword string is like "阿里巴巴 20\n腾讯 15"
            lines = hotword.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        fst_dict[" ".join(parts[:-1])] = int(parts[-1])
                    except ValueError:
                        log.warning(f"Skipping invalid hotword line: {line}")
                elif line.strip(): # Check if line is not empty after stripping
                     log.warning(f"Skipping invalid hotword line format: {line}")
            if fst_dict:
                hotword_msg = json.dumps(fst_dict)
                log.info(f"Using hotwords: {hotword_msg}")
            else:
                log.warning(f"No valid hotwords parsed from input: {hotword}")
        except Exception as e:
            log.error(f"Error processing hotwords: {e}")
            hotword_msg = "" # Fallback to empty if parsing fails

    # Determine WebSocket URI and SSL context
    if ssl_enabled:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        uri = f"wss://{host}:{port}"
    else:
        uri = f"ws://{host}:{port}"
        ssl_context = None

    log.info(f"Connecting to {uri} with mode: {mode}")

    try:
        # Increased max_size for potentially large messages/audio chunks
        async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context, max_size=None) as websocket:
            # --- Sending Task ---
            async def send_audio():
                try:
                    # Send initial configuration
                    config_message = json.dumps({
                        "mode": mode,
                        "chunk_size": chunk_size,
                        "chunk_interval": chunk_interval,
                        "audio_fs": audio_fs,
                        "wav_name": wav_name,
                        "wav_format": "pcm", # Assume raw bytes are PCM compatible for now
                        "is_speaking": True,
                        "hotwords": hotword_msg,
                        "itn": use_itn,
                    })
                    log.debug(f"Sending config: {config_message}")
                    await websocket.send(config_message)

                    # Send audio data in chunks
                    # Calculate stride based on the 'online' chunk size (middle value)
                    # Ensure sample rate and interval make sense
                    if audio_fs <= 0 or chunk_interval <= 0:
                        raise ValueError(f"Invalid audio_fs ({audio_fs}) or chunk_interval ({chunk_interval})")
                    stride = int(60 * chunk_size[1] / chunk_interval / 1000 * audio_fs * 2) # *2 for 16-bit audio
                    if stride <= 0:
                        log.warning(f"Calculated stride is {stride}. Sending audio in one chunk.")
                        stride = len(audio_bytes) # Avoid division by zero or sending nothing if params are weird

                    if not audio_bytes:
                         log.warning("Audio data is empty. Sending only config and closing message.")
                         chunk_num = 0
                    elif stride <= 0: # Handle case where audio exists but stride calculation failed
                        log.warning("Stride is zero or negative but audio exists. Sending as single chunk.")
                        chunk_num = 1
                        stride = len(audio_bytes)
                    else:
                        chunk_num = (len(audio_bytes) - 1) // stride + 1

                    log.info(f"Audio size: {len(audio_bytes)} bytes, Stride: {stride}, Chunks: {chunk_num}")

                    is_speaking = True
                    for i in range(chunk_num):
                        beg = i * stride
                        end = min(beg + stride, len(audio_bytes))
                        data = audio_bytes[beg:end]
                        if not data: # Should not happen with correct logic, but safety check
                            log.warning(f"Skipping empty chunk {i+1}/{chunk_num}")
                            continue

                        log.debug(f"Sending chunk {i+1}/{chunk_num}, size: {len(data)}")
                        await websocket.send(data)

                        # Determine sleep duration
                        # For offline mode, we can send faster. For online/2pass, simulate real-time.
                        sleep_duration = 0.001 if mode == "offline" else (60 * chunk_size[1] / chunk_interval / 1000)
                        # Add a small safety margin to sleep to avoid overwhelming the server
                        await asyncio.sleep(max(0.001, sleep_duration * 0.9)) # Sleep slightly less than interval

                    # Send final message after all chunks (or if no chunks)
                    is_speaking = False
                    final_chunk_message = json.dumps({"is_speaking": is_speaking})
                    log.debug(f"Sending final chunk marker: {final_chunk_message}")
                    await websocket.send(final_chunk_message)

                    log.info("Finished sending audio data.")

                except Exception as e:
                    log.error(f"Error during send_audio task: {e}")
                    log.error(traceback.format_exc())
                    # Signal receiver to stop waiting if sender fails critically
                    if not final_message_received.is_set():
                       final_message_received.set()
                    # Re-raise might be useful depending on how you want to handle API errors
                    # raise

            # --- Receiving Task ---
            async def receive_messages():
                nonlocal result
                try:
                    while True:
                        meg_str = await websocket.recv()
                        log.debug(f"Received message: {meg_str}")
                        try:
                            meg = json.loads(meg_str)
                        except json.JSONDecodeError:
                            log.error(f"Failed to decode JSON from server: {meg_str}")
                            continue # Skip this message

                        rec_mode = meg.get("mode", "unknown")
                        is_final = meg.get("is_final", False)

                        # Store the latest message, prioritize final messages for offline/2pass
                        if mode == "offline":
                            result = meg # Offline returns one final result
                            final_message_received.set()
                            log.info(f"Offline result received: {result.get('text')}")
                            break # Stop listening after getting the offline result
                        elif mode == "2pass":
                            # Assume the final message for 2pass also has 'is_final: true'
                            if is_final:
                                result = meg
                                final_message_received.set()
                                log.info(f"2Pass final result received: {result.get('text')}")
                                break # Stop listening after getting the final 2pass result
                            else:
                                log.debug(f"Ignoring intermediate 2pass message: {meg.get('text')}")
                                # Optionally store intermediate results if needed later
                        elif mode == "online":
                             # Online mode continuously sends results. For an API call,
                             # we likely want the *last* full result or accumulated.
                             # Let's store the latest one. The 'is_final' flag from server matters.
                             result = meg # Keep updating with the latest message
                             if is_final:
                                final_message_received.set()
                                log.info(f"Online final result received: {result.get('text')}")
                                break # Stop if server indicates final message
                             # We don't automatically stop for online unless the server closes
                             # or we implement a timeout/specific stop condition, or is_final is True.
                        else:
                             log.warning(f"Received message for unknown mode configuration: {rec_mode}")
                             result = meg # Store it anyway
                             if is_final: # Still respect is_final if server sends it
                                 final_message_received.set()
                                 break

                        # Connection state check is handled by exceptions below or server closing loop

                except websockets.exceptions.ConnectionClosedOK:
                    log.info("WebSocket connection closed gracefully by server.")
                except websockets.exceptions.ConnectionClosedError as e:
                    log.error(f"WebSocket connection closed with error: {e}")
                    # Don't raise HTTPException here, let the outer handler do it
                    # Set the event so the main wait doesn't time out unnecessarily if connection drops
                    if not final_message_received.is_set(): final_message_received.set()
                except Exception as e:
                    log.error(f"Error receiving message: {e}")
                    log.error(traceback.format_exc())
                    # Set the event so the main wait doesn't time out unnecessarily
                    if not final_message_received.is_set(): final_message_received.set()
                finally:
                    # Ensure the event is set if the loop exits for any reason other than explicit set
                    if not final_message_received.is_set():
                        log.warning("Receive loop finished without explicit final message signal.")
                        final_message_received.set()

            # --- Run Send and Receive Concurrently ---
            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_messages())

            # Wait for the final message signal, with a timeout
            timeout_seconds = 180 # Adjust as needed, depends on audio length & processing time
            try:
                log.debug(f"Waiting for final message signal with timeout {timeout_seconds}s")
                await asyncio.wait_for(final_message_received.wait(), timeout=timeout_seconds)
                log.info("Final message signal received or receive task ended.")
            except asyncio.TimeoutError:
                 log.error(f"Timeout waiting for transcription result after {timeout_seconds}s.")
                 # Attempt to cancel tasks if they are still running
                 if not send_task.done(): send_task.cancel()
                 if not receive_task.done(): receive_task.cancel()
                 # Wait briefly for cancellation to be processed
                 await asyncio.gather(send_task, receive_task, return_exceptions=True)
                 raise HTTPException(status_code=504, detail="Timeout waiting for FunASR result")

            # Ensure both tasks have completed and check for exceptions
            await asyncio.gather(send_task, receive_task, return_exceptions=True)

            # Check if tasks raised exceptions that weren't handled internally
            if send_task.done() and send_task.exception():
                 log.error(f"Send task finished with exception: {send_task.exception()}")
                 # Decide if this should be an HTTP error
                 # raise HTTPException(status_code=500, detail=f"Error sending audio: {send_task.exception()}")
            if receive_task.done() and receive_task.exception():
                 # Errors inside receive_messages should ideally be handled there or are connection errors
                 log.error(f"Receive task finished with exception: {receive_task.exception()}")
                 # Don't raise here if it was just a connection closed error handled previously

    except websockets.exceptions.InvalidURI:
        log.error(f"Invalid WebSocket URI: {uri}")
        raise HTTPException(status_code=400, detail="Invalid WebSocket URI format")
    except websockets.exceptions.WebSocketException as e:
        log.error(f"WebSocket connection failed: {e}")
        raise HTTPException(status_code=502, detail=f"Could not connect to FunASR WebSocket: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred during WebSocket interaction: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    if result:
        return result
    else:
        # This case might happen if the connection closed before *any* result was received,
        # or if the timeout occurred before 'result' was assigned, or receiver failed badly.
        log.error("No valid result received from FunASR service.")
        # Check websocket status again? Maybe not necessary if exceptions were caught.
        raise HTTPException(status_code=502, detail="No result received from FunASR service (check logs for connection/processing errors).")


# --- API Endpoint Definition ---
@app.post("/transcribe", response_class=JSONResponse)
async def http_transcribe(
    # --- FunASR Parameters (as Query or Form data) ---
    host: str = Query("127.0.0.1", description="FunASR WebSocket server host IP."),
    port: int = Query(10098, description="FunASR WebSocket server port."),
    # --- CORRECTED: Use pattern instead of regex ---
    mode: str = Query("offline", description="Recognition mode: offline, online, 2pass.", pattern="^(offline|online|2pass)$"),
    chunk_size_str: str = Query("5,10,5", alias="chunk_size", description="Chunk size tuple (comma-separated): 'before,middle,after' e.g., 5,10,5."),
    chunk_interval: int = Query(10, description="Chunk processing interval in ms."),
    hotword: Optional[str] = Form(None, description="Hotwords string, one per line (e.g., '阿里巴巴 20\\n腾讯 15'). URL Encode if needed."),
    audio_fs: int = Query(16000, description="Audio sample rate in Hz."),
    ssl_enabled: bool = Query(False, alias="ssl", description="Use wss:// secure connection."),
    use_itn: bool = Query(True, description="Apply Inverse Text Normalization."),
    # --- Audio Input (as File Upload) ---
    audio_file: UploadFile = File(..., description="Audio file to transcribe (e.g., wav, pcm)."),
):
    """
    Receives an audio file and parameters, interacts with the FunASR WebSocket service,
    and returns the transcription result as JSON.
    """
    start_time = time.time()
    log.info(f"Received request for file: {audio_file.filename}, mode: {mode}")

    # --- Parameter Parsing and Validation ---
    try:
        chunk_size = [int(x.strip()) for x in chunk_size_str.split(",")]
        if len(chunk_size) != 3:
            raise ValueError("chunk_size must have 3 comma-separated integers.")
    except ValueError as e:
        log.error(f"Invalid chunk_size format: {chunk_size_str}. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid chunk_size format: '{chunk_size_str}'. Must be 'int,int,int'.")

    # Read audio file content
    try:
        audio_bytes = await audio_file.read()
        wav_name = audio_file.filename if audio_file.filename else "uploaded_audio"
        log.info(f"Read {len(audio_bytes)} bytes from {wav_name}")
        if not audio_bytes:
             raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    except Exception as e:
        log.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")
    finally:
        await audio_file.close() # Ensure file handle is closed


    # --- Call the WebSocket Interaction Logic ---
    try:
        transcription_result = await transcribe_audio_ws(
            host=host,
            port=port,
            chunk_size=chunk_size,
            chunk_interval=chunk_interval,
            hotword=hotword if hotword else "", # Pass empty string if None
            audio_bytes=audio_bytes,
            audio_fs=audio_fs,
            wav_name=wav_name,
            ssl_enabled=ssl_enabled,
            use_itn=use_itn,
            mode=mode,
        )
        end_time = time.time()
        log.info(f"Successfully transcribed {wav_name} in {end_time - start_time:.2f} seconds")
        # Return the entire JSON message received from FunASR
        return JSONResponse(content=transcription_result)

    except HTTPException as e:
        # Re-raise HTTPExceptions that occurred during WebSocket interaction or validation
        log.warning(f"Transcription failed for {wav_name} with HTTPException: {e.status_code} - {e.detail}")
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the process
        end_time = time.time()
        log.error(f"Error during transcription process for {wav_name} after {end_time - start_time:.2f} seconds: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error during transcription: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Use arguments to configure the API server itself (optional)
    parser = argparse.ArgumentParser(description="Run FunASR HTTP API Server")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="Host for the API server")
    parser.add_argument("--api_port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers (for production)")
    api_args = parser.parse_args()

    print(f"Starting FunASR API Server on http://{api_args.api_host}:{api_args.api_port}")

    # --- CORRECTED: Use "main:app" to refer to the app object in *this* file ---
    uvicorn.run(
        "main:app",  # Tells uvicorn: find the object 'app' in the module 'main.py'
        host=api_args.api_host,
        port=api_args.api_port,
        workers=api_args.workers, # For production deployment, consider > 1 worker if appropriate
        # reload=True # Enable reload for development ONLY if needed, disable for production
                      # Uvicorn needs to be installed with 'standard' extras for reload: pip install uvicorn[standard]
    )

#  python main.py --api_host 0.0.0.0 --api_port 8000
#  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 10 --log-config log_config.yaml
# curl -X POST "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&ssl=false&use_itn=true&audio_fs=16000" \
#      -H "accept: application/json" \
#      -F "audio_file=@../audio/asr_example.wav" # Adjust path as needed

# {"is_final":false,"mode":"offline","stamp_sents":[{"end":5195,"punc":"。","start":880,"text_seg":"欢 迎 大 家 来 体 验 达 摩 院 推 出 的 语 音 识 别 模 型","ts_list":[[880,1120],[1120,1380],[1380,1540],[1540,1780],[1780,2020],[2020,2180],[2180,2500],[2500,2659],[2659,2780],[2780,3040],[3040,3240],[3240,3480],[3480,3699],[3699,3900],[3900,4180],[4180,4420],[4420,4620],[4620,4780],[4780,5195]]}],"text":"欢迎大家来体验达摩院推出的语音识别模型。","timestamp":"[[880,1120],[1120,1380],[1380,1540],[1540,1780],[1780,2020],[2020,2180],[2180,2500],[2500,2659],[2659,2780],[2780,3040],[3040,3240],[3240,3480],[3480,3699],[3699,3900],[3900,4180],[4180,4420],[4420,4620],[4620,4780],[4780,5195]]","wav_name":"asr_example.wav"}

# 有一个ASR接口参考curl调用：
# curl -X POST "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&ssl=false&use_itn=true&audio_fs=16000" \
#      -H "accept: application/json" \
#      -F "audio_file=@../audio/asr_example.wav" # Adjust path as needed
# 后可获得解析结果：
# {"is_final":false,"mode":"offline","stamp_sents":[{"end":6550,"punc":"。","start":250,"text_seg":"我 们 注 意 到 您 近 期 咨 询 了 关 于 我 行 星 推 出 的 一 款 结 构 性 理 财 产 品","ts_list":[[250,350],[350,490],[490,630],[630,810],[810,1130],[1130,1330],[1330,1530],[1530,1850],[1850,2010],[2010,2250],[2250,2570],[2570,2730],[2730,2970],[2970,3170],[3170,3390],[3390,3670],[3670,3870],[3870,4050],[4050,4210],[4210,4350],[4350,4670],[4670,4830],[4830,5050],[5050,5330],[5330,5470],[5470,5690],[5690,5950],[5950,6550]]},{"end":9190,"punc":"，","start":6610,"text_seg":"这 款 产 品 本 金 相 对 安 全","ts_list":[[6610,6790],[6790,6990],[6990,7150],[7150,7490],[7490,7690],[7690,7990],[7990,8170],[8170,8389],[8389,8590],[8590,9190]]},{"end":12530,"punc":"，","start":9270,"text_seg":"其 预 期 收 益 与 特 定 的 市 场 指 标","ts_list":[[9270,9570],[9570,9730],[9730,9929],[9929,10210],[10210,10530],[10530,10770],[10770,10969],[10969,11210],[11210,11349],[11349,11450],[11450,11730],[11730,11929],[11929,12530]]},{"end":15790,"punc":"，","start":12530,"text_seg":"例 如 某 个 股 票 指 数 或 汇 率 挂 钩","ts_list":[[12530,12770],[12770,13030],[13030,13190],[13190,13389],[13389,13549],[13549,13809],[13809,13969],[13969,14349],[14349,14670],[14670,14929],[14929,15190],[15190,15389],[15389,15790]]},{"end":16949,"punc":"。","start":15790,"text_seg":"请 您 知 悉","ts_list":[[15790,15990],[15990,16190],[16190,16350],[16350,16949]]},{"end":19975,"punc":"，","start":16970,"text_seg":"虽 然 它 提 供 了 潜 在 的 较 高 回 报 机 会","ts_list":[[16970,17130],[17130,17310],[17310,17529],[17529,17730],[17730,17910],[17910,18109],[18109,18390],[18390,18630],[18630,18789],[18789,18990],[18990,19210],[19210,19410],[19410,19630],[19630,19830],[19830,19975]]},{"end":23640,"punc":"，","start":20440,"text_seg":"但 实 际 收 益 可 能 存 在 不 确 定 性","ts_list":[[20440,20680],[20680,20880],[20880,21140],[21140,21300],[21300,21640],[21640,21780],[21780,22000],[22000,22260],[22260,22460],[22460,22640],[22640,22860],[22860,23100],[23100,23640]]},{"end":29420,"punc":"。","start":23640,"text_seg":"极 端 情 况 下 可 能 仅 能 获 得 较 低 的 保 底 收 益 甚 至 零 收 益","ts_list":[[23640,23820],[23820,24020],[24020,24160],[24160,24400],[24400,24740],[24740,24940],[24940,25180],[25180,25360],[25360,25560],[25560,25780],[25780,26060],[26060,26260],[26260,26500],[26500,26700],[26700,26900],[26900,27160],[27160,27380],[27380,27860],[27860,28100],[28100,28420],[28420,28660],[28660,28880],[28880,29420]]},{"end":31040,"punc":"，","start":29420,"text_seg":"请 在 投 资 前 确 认","ts_list":[[29420,29600],[29600,29900],[29900,30080],[30080,30320],[30320,30660],[30660,30840],[30840,31040]]},{"end":33805,"punc":"。","start":31040,"text_seg":"以 充 分 理 解 产 品 结 构 和 风 险","ts_list":[[31040,31280],[31280,31439],[31439,31660],[31660,31860],[31860,32060],[32060,32279],[32279,32540],[32540,32760],[32760,33059],[33059,33260],[33260,33480],[33480,33805]]}],"text":"我们注意到您近期咨询了关于我行星推出的一款结构性理财产品。这款产品本金相对安全，其预期收益与特定的市场指标，例如某个股票指数或汇率挂钩，请您知悉。虽然它提供了潜在的较高回报机会，但实际收益可能存在不确定性，极端情况下可能仅能获得较低的保底收益甚至零收益。请在投资前确认，以充分理解产品结构和风险。","timestamp":"[[250,350],[350,490],[490,630],[630,810],[810,1130],[1130,1330],[1330,1530],[1530,1850],[1850,2010],[2010,2250],[2250,2570],[2570,2730],[2730,2970],[2970,3170],[3170,3390],[3390,3670],[3670,3870],[3870,4050],[4050,4210],[4210,4350],[4350,4670],[4670,4830],[4830,5050],[5050,5330],[5330,5470],[5470,5690],[5690,5950],[5950,6550],[6610,6790],[6790,6990],[6990,7150],[7150,7490],[7490,7690],[7690,7990],[7990,8170],[8170,8389],[8389,8590],[8590,9190],[9270,9570],[9570,9730],[9730,9929],[9929,10210],[10210,10530],[10530,10770],[10770,10969],[10969,11210],[11210,11349],[11349,11450],[11450,11730],[11730,11929],[11929,12530],[12530,12770],[12770,13030],[13030,13190],[13190,13389],[13389,13549],[13549,13809],[13809,13969],[13969,14349],[14349,14670],[14670,14929],[14929,15190],[15190,15389],[15389,15790],[15790,15990],[15990,16190],[16190,16350],[16350,16949],[16970,17130],[17130,17310],[17310,17529],[17529,17730],[17730,17910],[17910,18109],[18109,18390],[18390,18630],[18630,18789],[18789,18990],[18990,19210],[19210,19410],[19410,19630],[19630,19830],[19830,19975],[20440,20680],[20680,20880],[20880,21140],[21140,21300],[21300,21640],[21640,21780],[21780,22000],[22000,22260],[22260,22460],[22460,22640],[22640,22860],[22860,23100],[23100,23640],[23640,23820],[23820,24020],[24020,24160],[24160,24400],[24400,24740],[24740,24940],[24940,25180],[25180,25360],[25360,25560],[25560,25780],[25780,26060],[26060,26260],[26260,26500],[26500,26700],[26700,26900],[26900,27160],[27160,27380],[27380,27860],[27860,28100],[28100,28420],[28420,28660],[28660,28880],[28880,29420],[29420,29600],[29600,29900],[29900,30080],[30080,30320],[30320,30660],[30660,30840],[30840,31040],[31040,31280],[31280,31439],[31439,31660],[31660,31860],[31860,32060],[32060,32279],[32279,32540],[32540,32760],[32760,33059],[33059,33260],[33260,33480],[33480,33805]]","wav_name":"sentence_104.wav"}

# 但是上述都是基于单声道的，如果在应用层做适配，对于双声道音频拆分左右声道，也就是角色A和B。对于拆分之后的音频调用接口（同时发起调用），获取返回结果，并根据返回结果组织语句顺序，给出组合之后的ASR输出。基于高可用和高并发的特点，给出python形式提供api的完整的代码，并给出测试curl方法。