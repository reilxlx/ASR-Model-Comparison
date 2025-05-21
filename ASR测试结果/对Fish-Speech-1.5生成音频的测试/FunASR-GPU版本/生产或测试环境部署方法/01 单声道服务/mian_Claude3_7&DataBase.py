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
import socket
import uuid
import datetime
import pymysql
import pymysql.cursors
from pathlib import Path

# --- API Specific Imports ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Header
from fastapi.responses import JSONResponse
import aiofiles # For async file reading and writing
from typing import List, Optional, Dict, Any

# 设置日志级别
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- 数据库连接配置 ---
DB_CONFIG = {
    'host': '***********.sql.tencentcdb.com',
    'port': *****,
    'user': '*****',
    'password': '*******',
    'database': '*****',  # 实际数据库名称
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# --- 临时音频文件存储目录 ---
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- FastAPI App初始化 ---
app = FastAPI(
    title="FunASR HTTP API",
    description="FunASR WebSocket流式客户端的API封装，支持数据库记录功能。",
)

# --- JSON字符串处理函数 ---
def chinese_json_dumps(obj):
    """确保JSON字符串中的中文不会被转义为Unicode"""
    return json.dumps(obj, ensure_ascii=False)

# --- 获取服务器IP地址 ---
def get_server_ip():
    try:
        # 创建一个临时套接字连接，用于获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        log.warning(f"获取服务器IP失败: {e}，使用默认IP 127.0.0.1")
        return "127.0.0.1"

# --- 数据库操作函数 ---
async def save_to_database(data):
    """保存处理数据到数据库"""
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
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
        with conn.cursor() as cursor:
            cursor.execute(sql, data)
        conn.commit()
        log.info(f"成功将请求记录保存到数据库，REQ_SEQNO: {data.get('REQ_SEQNO')}")
        return True
    except Exception as e:
        log.error(f"数据库操作失败: {e}")
        log.error(traceback.format_exc())
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

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
    req_seqno: str = None,  # 新增参数：请求序列号
    save_wav: bool = True,  # 新增参数：是否保存音频文件
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

# --- 保存音频文件函数 ---
async def save_audio_file(audio_bytes: bytes, filename: str):
    """保存音频数据到临时目录"""
    try:
        file_path = os.path.join(TEMP_AUDIO_DIR, filename)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(audio_bytes)
        log.info(f"音频文件已保存至: {file_path}")
        return file_path
    except Exception as e:
        log.error(f"保存音频文件失败: {e}")
        log.error(traceback.format_exc())
        return None

# --- 生成请求序列号 ---
def generate_req_seqno():
    """生成唯一的请求序列号"""
    # 格式：ASR + 8位日期 + UUID前8位
    today = datetime.datetime.now().strftime('%Y%m%d')
    uuid_part = str(uuid.uuid4()).replace('-', '')[:8]
    return f"ASR{today}{uuid_part}"

# --- 双声道音频处理函数 ---
def split_stereo_audio(audio_bytes, wav_name, audio_fs=16000):
    """
    将双声道音频分割为左右声道，返回两个单独的字节数组
    """
    try:
        import numpy as np
        import io
        import wave
        
        # 判断是否为WAV格式
        is_wav = False
        if wav_name.lower().endswith('.wav'):
            is_wav = True
            
        if is_wav:
            # 使用wave模块读取WAV
            with io.BytesIO(audio_bytes) as wav_file:
                with wave.open(wav_file, 'rb') as wav:
                    channels = wav.getnchannels()
                    if channels != 2:
                        log.warning(f"音频文件 {wav_name} 不是双声道, 实际声道数: {channels}")
                        return None, None
                    
                    # 读取所有采样
                    frames = wav.readframes(wav.getnframes())
                    sample_width = wav.getsampwidth()
                    
                    # 转换为NumPy数组
                    if sample_width == 2:  # 16-bit
                        data = np.frombuffer(frames, dtype=np.int16)
                    elif sample_width == 4:  # 32-bit
                        data = np.frombuffer(frames, dtype=np.int32)
                    else:
                        log.error(f"不支持的采样位宽: {sample_width * 8}位")
                        return None, None
                    
                    # 重塑为[n_samples, 2]的数组
                    data = data.reshape(-1, 2)
                    
                    # 拆分声道
                    left_channel = data[:, 0]
                    right_channel = data[:, 1]
                    
                    # 转回字节
                    left_bytes = left_channel.tobytes()
                    right_bytes = right_channel.tobytes()
                    
                    # 创建单声道WAV文件
                    left_wav_bytes = io.BytesIO()
                    right_wav_bytes = io.BytesIO()
                    
                    # 写入左声道WAV
                    with wave.open(left_wav_bytes, 'wb') as left_wav:
                        left_wav.setnchannels(1)
                        left_wav.setsampwidth(sample_width)
                        left_wav.setframerate(wav.getframerate())
                        left_wav.writeframes(left_bytes)
                    
                    # 写入右声道WAV
                    with wave.open(right_wav_bytes, 'wb') as right_wav:
                        right_wav.setnchannels(1)
                        right_wav.setsampwidth(sample_width)
                        right_wav.setframerate(wav.getframerate())
                        right_wav.writeframes(right_bytes)
                    
                    return left_wav_bytes.getvalue(), right_wav_bytes.getvalue()
        else:
            # 假设是PCM原始数据 (需要知道采样率和位宽)
            # 通常采用16位采样
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(data) % 2 != 0:
                log.warning(f"PCM数据长度不是偶数，可能不是双声道或格式错误")
                return None, None
                
            # 重塑为[n_samples, 2]的数组
            data = data.reshape(-1, 2)
            
            # 拆分声道
            left_channel = data[:, 0]
            right_channel = data[:, 1]
            
            # 转回字节
            left_bytes = left_channel.tobytes()
            right_bytes = right_channel.tobytes()
            
            # 创建单声道WAV文件
            left_wav_bytes = io.BytesIO()
            right_wav_bytes = io.BytesIO()
            
            # 写入左声道WAV
            with wave.open(left_wav_bytes, 'wb') as left_wav:
                left_wav.setnchannels(1)
                left_wav.setsampwidth(2)  # 16-bit
                left_wav.setframerate(audio_fs)
                left_wav.writeframes(left_bytes)
            
            # 写入右声道WAV
            with wave.open(right_wav_bytes, 'wb') as right_wav:
                right_wav.setnchannels(1)
                right_wav.setsampwidth(2)  # 16-bit
                right_wav.setframerate(audio_fs)
                right_wav.writeframes(right_bytes)
            
            return left_wav_bytes.getvalue(), right_wav_bytes.getvalue()
            
    except Exception as e:
        log.error(f"拆分双声道音频失败: {e}")
        log.error(traceback.format_exc())
        return None, None


# --- 组合时间戳排序函数 ---
def merge_transcriptions_by_timestamp(channel_a_result, channel_b_result):
    """
    根据时间戳合并两个转录结果，按时间顺序排序
    """
    try:
        # 解析结果
        if isinstance(channel_a_result, str):
            channel_a_result = json.loads(channel_a_result)
        if isinstance(channel_b_result, str):
            channel_b_result = json.loads(channel_b_result)
            
        # 提取句子和时间戳
        sentences_a = []
        if 'stamp_sents' in channel_a_result:
            for sent in channel_a_result['stamp_sents']:
                sentences_a.append({
                    'text': sent['text_seg'].replace(' ', ''),
                    'start': sent['start'],
                    'end': sent['end'],
                    'channel': 'A',
                    'punc': sent.get('punc', '')
                })
        
        sentences_b = []
        if 'stamp_sents' in channel_b_result:
            for sent in channel_b_result['stamp_sents']:
                sentences_b.append({
                    'text': sent['text_seg'].replace(' ', ''),
                    'start': sent['start'],
                    'end': sent['end'],
                    'channel': 'B',
                    'punc': sent.get('punc', '')
                })
                
        # 合并并排序
        all_sentences = sentences_a + sentences_b
        sorted_sentences = sorted(all_sentences, key=lambda x: x['start'])
        
        # 创建合并结果
        merged_text = ""
        merged_sentences = []
        
        for sent in sorted_sentences:
            speaker = f"[{sent['channel']}]"
            sentence_text = f"{speaker}{sent['text']}{sent['punc']} "
            merged_text += sentence_text
            
            merged_sentences.append({
                'text': sent['text'],
                'start': sent['start'],
                'end': sent['end'],
                'channel': sent['channel'],
                'punc': sent['punc']
            })
            
        return {
            'text': merged_text.strip(),
            'mode': 'stereo',
            'is_final': True,
            'stamp_sents': merged_sentences,
            'channel_a': channel_a_result.get('text', ''),
            'channel_b': channel_b_result.get('text', ''),
            'wav_name': channel_a_result.get('wav_name', '') + '_stereo'
        }
            
    except Exception as e:
        log.error(f"合并转录结果失败: {e}")
        log.error(traceback.format_exc())
        # 返回部分结果
        return {
            'text': f"[A]{channel_a_result.get('text', '')} [B]{channel_b_result.get('text', '')}",
            'mode': 'stereo',
            'is_final': True,
            'error': f"合并转录结果失败: {str(e)}",
            'channel_a': channel_a_result.get('text', ''),
            'channel_b': channel_b_result.get('text', '')
        }


# --- 双声道API端点 ---
@app.post("/transcribe_stereo", response_class=JSONResponse)
async def transcribe_stereo(
    # --- FunASR参数 ---
    host: str = Query("127.0.0.1", description="FunASR WebSocket服务器主机IP"),
    port: int = Query(10098, description="FunASR WebSocket服务器端口"),
    mode: str = Query("offline", description="识别模式: offline, online, 2pass", pattern="^(offline|online|2pass)$"),
    chunk_size_str: str = Query("5,10,5", alias="chunk_size", description="块大小元组(逗号分隔): 'before,middle,after'，例如5,10,5"),
    chunk_interval: int = Query(10, description="块处理间隔(毫秒)"),
    hotword: Optional[str] = Form(None, description="热词字符串，每行一个(例如，'阿里巴巴 20\\n腾讯 15')，需要URL编码"),
    audio_fs: int = Query(16000, description="音频采样率(Hz)"),
    ssl_enabled: bool = Query(False, alias="ssl", description="使用wss://安全连接"),
    use_itn: bool = Query(True, description="应用文本反规范化(ITN)"),
    
    # --- 音频输入 ---
    audio_file: UploadFile = File(..., description="要转录的双声道音频文件(例如wav, pcm)"),
    
    # --- 数据库记录相关参数 ---
    req_seqno: Optional[str] = Query(None, description="请求序列号，如不提供则自动生成"),
    req_channelid: str = Query("API", description="请求通道ID"),
    req_bank_code: Optional[str] = Query(None, description="请求银行代码"),
    req_userid: Optional[str] = Query(None, description="请求用户ID"),
    req_serviceid: str = Query("ASRSERVICE", description="请求服务ID"),
    req_message: Optional[str] = Form(None, description="请求消息"),
    req_document: Optional[str] = Form(None, description="请求文档"),
    save_wav: bool = Query(True, description="是否保存音频文件"),
    
    # --- 系统生成的其他信息 ---
    user_agent: Optional[str] = Header(None, description="用户代理")
):
    """
    接收双声道音频文件，拆分为左右声道，与FunASR WebSocket服务交互，
    处理两个转录结果并根据时间戳合并它们。同时将请求和结果记录到数据库中。
    """
    # 生成或使用提供的请求序列号
    if not req_seqno:
        req_seqno = generate_req_seqno()

    start_time = time.time()
    log.info(f"收到双声道转录请求，文件: {audio_file.filename}, 模式: {mode}, 序列号: {req_seqno}")
    
    # 生成交易日期和时间戳
    tran_date = datetime.datetime.now().strftime('%Y%m%d')
    tran_timestamp = datetime.datetime.now().strftime('%H%M%S%f')[:9]
    
    # 获取服务器IP
    server_ip = get_server_ip()
    
    # 保存原始音频文件路径(如果save_wav为True)
    stereo_wav_path = None
    
    # --- 读取音频文件内容 ---
    try:
        audio_bytes = await audio_file.read()
        wav_name = audio_file.filename if audio_file.filename else f"uploaded_stereo_{req_seqno}"
        log.info(f"读取了 {len(audio_bytes)} 字节，来自 {wav_name}")
        
        if not audio_bytes:
            error_msg = "上传的音频文件为空。"
            db_record = {
                'TRAN_DATE': tran_date,
                'TRAN_TIMESTAMP': tran_timestamp,
                'REQ_SEQNO': req_seqno,
                'REQ_CHANNELID': req_channelid,
                'REQ_BANK_CODE': req_bank_code,
                'REQ_USERID': req_userid,
                'REQ_SERVICEID': req_serviceid,
                'REQ_MESSAGE': chinese_json_dumps({"user_agent": user_agent, "stereo": True}) if user_agent else None,
                'REQ_DOCUMENT': req_document,
                'RES_ERROR_CODE': '400',
                'RES_ERROR_MESSAGE': 'Empty Audio',
                'RES_DOCUMENT': None,
                'RES_MESSAGE': chinese_json_dumps({"error": error_msg}),
                'SERVER_IP': server_ip,
                'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
                'ASR_FIND_WAV_PATH_STATUS': 'N',
                'ASR_VOICERESULT': None,
                'ASR_WAV_PATH': None
            }
            await save_to_database(db_record)
            raise HTTPException(status_code=400, detail=error_msg)
            
        # 如果需要，保存原始双声道音频文件
        if save_wav:
            # 添加序列号到文件名以确保唯一性
            file_ext = os.path.splitext(wav_name)[1] if '.' in wav_name else '.wav'
            unique_filename = f"{os.path.splitext(wav_name)[0]}_{req_seqno}_stereo{file_ext}"
            stereo_wav_path = await save_audio_file(audio_bytes, unique_filename)
            
        # 拆分双声道音频
        left_bytes, right_bytes = split_stereo_audio(audio_bytes, wav_name, audio_fs)
        
        if left_bytes is None or right_bytes is None:
            error_msg = "无法拆分双声道音频，可能不是双声道格式或处理失败。"
            db_record = {
                'TRAN_DATE': tran_date,
                'TRAN_TIMESTAMP': tran_timestamp,
                'REQ_SEQNO': req_seqno,
                'REQ_CHANNELID': req_channelid,
                'REQ_BANK_CODE': req_bank_code,
                'REQ_USERID': req_userid,
                'REQ_SERVICEID': req_serviceid,
                'REQ_MESSAGE': chinese_json_dumps({"user_agent": user_agent, "stereo": True}) if user_agent else None,
                'REQ_DOCUMENT': req_document,
                'RES_ERROR_CODE': '400',
                'RES_ERROR_MESSAGE': 'Split Stereo Failed',
                'RES_DOCUMENT': None,
                'RES_MESSAGE': chinese_json_dumps({"error": error_msg}),
                'SERVER_IP': server_ip,
                'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
                'ASR_FIND_WAV_PATH_STATUS': 'Y' if stereo_wav_path else 'N',
                'ASR_VOICERESULT': None,
                'ASR_WAV_PATH': stereo_wav_path
            }
            await save_to_database(db_record)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 保存拆分后的单声道文件
        left_wav_path = None
        right_wav_path = None
        
        if save_wav:
            file_base = os.path.splitext(wav_name)[0]
            file_ext = os.path.splitext(wav_name)[1] if '.' in wav_name else '.wav'
            left_filename = f"{file_base}_{req_seqno}_left{file_ext}"
            right_filename = f"{file_base}_{req_seqno}_right{file_ext}"
            
            left_wav_path = await save_audio_file(left_bytes, left_filename)
            right_wav_path = await save_audio_file(right_bytes, right_filename)
        
        # 为左右声道创建唯一的子序列号
        channel_a_seqno = f"{req_seqno}_A"
        channel_b_seqno = f"{req_seqno}_B"
        
        # 创建左右声道文件名
        left_wav_name = f"{os.path.splitext(wav_name)[0]}_left"
        right_wav_name = f"{os.path.splitext(wav_name)[0]}_right"
        
        # 并发处理左右声道
        # 创建任务
        channel_a_task = asyncio.create_task(
            transcribe_audio_ws(
                host=host,
                port=port,
                chunk_size=chunk_size,
                chunk_interval=chunk_interval,
                hotword=hotword if hotword else "",
                audio_bytes=left_bytes,
                audio_fs=audio_fs,
                wav_name=left_wav_name,
                ssl_enabled=ssl_enabled,
                use_itn=use_itn,
                mode=mode,
                req_seqno=channel_a_seqno,
                save_wav=False  # 已单独保存
            )
        )
        
        channel_b_task = asyncio.create_task(
            transcribe_audio_ws(
                host=host,
                port=port,
                chunk_size=chunk_size,
                chunk_interval=chunk_interval,
                hotword=hotword if hotword else "",
                audio_bytes=right_bytes,
                audio_fs=audio_fs,
                wav_name=right_wav_name,
                ssl_enabled=ssl_enabled,
                use_itn=use_itn,
                mode=mode,
                req_seqno=channel_b_seqno,
                save_wav=False  # 已单独保存
            )
        )
        
        # 等待两个任务完成
        channel_a_result, channel_b_result = await asyncio.gather(channel_a_task, channel_b_task, return_exceptions=True)
        
        # 检查任务是否成功
        if isinstance(channel_a_result, Exception):
            log.error(f"处理左声道失败: {channel_a_result}")
            channel_a_error = str(channel_a_result)
            channel_a_result = {"text": f"左声道处理失败: {channel_a_error}", "error": channel_a_error}
        
        if isinstance(channel_b_result, Exception):
            log.error(f"处理右声道失败: {channel_b_result}")
            channel_b_error = str(channel_b_result)
            channel_b_result = {"text": f"右声道处理失败: {channel_b_error}", "error": channel_b_error}
        
        # 合并结果
        merged_result = merge_transcriptions_by_timestamp(channel_a_result, channel_b_result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        log.info(f"成功转录双声道音频 {wav_name}，用时 {processing_time:.2f} 秒")
        
        # 记录成功结果到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({
                "mode": mode,
                "chunk_size": chunk_size_str,
                "chunk_interval": chunk_interval,
                "audio_fs": audio_fs,
                "use_itn": use_itn,
                "user_agent": user_agent,
                "hotword": hotword,
                "stereo": True
            }),
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': '0000',
            'RES_ERROR_MESSAGE': '成功',
            'RES_DOCUMENT': chinese_json_dumps({
                "channel_a_path": left_wav_path,
                "channel_b_path": right_wav_path
            }),
            'RES_MESSAGE': chinese_json_dumps({"responseCode": 200, "detail": "双声道转录成功"}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{processing_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'Y' if stereo_wav_path else 'N',
            'ASR_VOICERESULT': chinese_json_dumps(merged_result),
            'ASR_WAV_PATH': stereo_wav_path
        }
        
        await save_to_database(db_record)
        
        # 返回合并的结果
        return JSONResponse(content=merged_result)
        
    except HTTPException as e:
        # 重新抛出HTTP异常
        raise e
    except Exception as e:
        log.error(f"双声道转录过程中出错: {e}")
        log.error(traceback.format_exc())
        error_msg = f"双声道转录期间内部服务器错误: {e}"
        
        # 记录错误到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({
                "mode": mode,
                "chunk_size": chunk_size_str,
                "chunk_interval": chunk_interval,
                "audio_fs": audio_fs,
                "use_itn": use_itn,
                "user_agent": user_agent,
                "hotword": hotword,
                "stereo": True
            }),
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': '500',
            'RES_ERROR_MESSAGE': '内部服务器错误',
            'RES_DOCUMENT': None,
            'RES_MESSAGE': chinese_json_dumps({"error": error_msg}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'Y' if stereo_wav_path else 'N',
            'ASR_VOICERESULT': None,
            'ASR_WAV_PATH': stereo_wav_path
        }
        
        await save_to_database(db_record)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        await audio_file.close()  # 确保文件句柄被关闭


# --- API端点定义 ---
@app.post("/transcribe", response_class=JSONResponse)
async def http_transcribe(
    # --- FunASR参数 ---
    host: str = Query("127.0.0.1", description="FunASR WebSocket服务器主机IP"),
    port: int = Query(10098, description="FunASR WebSocket服务器端口"),
    mode: str = Query("offline", description="识别模式: offline, online, 2pass", pattern="^(offline|online|2pass)$"),
    chunk_size_str: str = Query("5,10,5", alias="chunk_size", description="块大小元组(逗号分隔): 'before,middle,after'，例如5,10,5"),
    chunk_interval: int = Query(10, description="块处理间隔(毫秒)"),
    hotword: Optional[str] = Form(None, description="热词字符串，每行一个(例如，'阿里巴巴 20\\n腾讯 15')，需要URL编码"),
    audio_fs: int = Query(16000, description="音频采样率(Hz)"),
    ssl_enabled: bool = Query(False, alias="ssl", description="使用wss://安全连接"),
    use_itn: bool = Query(True, description="应用文本反规范化(ITN)"),
    
    # --- 音频输入 ---
    audio_file: UploadFile = File(..., description="要转录的音频文件(例如wav, pcm)"),
    
    # --- 数据库记录相关参数 ---
    req_seqno: Optional[str] = Query(None, description="请求序列号，如不提供则自动生成"),
    req_channelid: str = Query("API", description="请求通道ID"),
    req_bank_code: Optional[str] = Query(None, description="请求银行代码"),
    req_userid: Optional[str] = Query(None, description="请求用户ID"),
    req_serviceid: str = Query("ASRSERVICE", description="请求服务ID"),
    req_message: Optional[str] = Form(None, description="请求消息"),
    req_document: Optional[str] = Form(None, description="请求文档"),
    save_wav: bool = Query(True, description="是否保存音频文件"),
    
    # --- 系统生成的其他信息 ---
    user_agent: Optional[str] = Header(None, description="用户代理")
):
    """
    接收音频文件和参数，与FunASR WebSocket服务交互，并返回转录结果作为JSON。
    同时将请求和结果记录到数据库中，并可选地保存音频文件。
    """
    # 生成或使用提供的请求序列号
    if not req_seqno:
        req_seqno = generate_req_seqno()

    start_time = time.time()
    log.info(f"收到请求，文件: {audio_file.filename}, 模式: {mode}, 序列号: {req_seqno}")
    
    # 生成交易日期和时间戳
    tran_date = datetime.datetime.now().strftime('%Y%m%d')
    tran_timestamp = datetime.datetime.now().strftime('%H%M%S%f')[:9]
    
    # 获取服务器IP
    server_ip = get_server_ip()
    
    # 保存音频文件路径(如果save_wav为True)
    wav_path = None
    
    # --- 解析参数和验证 ---
    try:
        chunk_size = [int(x.strip()) for x in chunk_size_str.split(",")]
        if len(chunk_size) != 3:
            raise ValueError("chunk_size必须有3个逗号分隔的整数。")
    except ValueError as e:
        log.error(f"无效的chunk_size格式: {chunk_size_str}. 错误: {e}")
        error_msg = f"无效的chunk_size格式: '{chunk_size_str}'。必须是'int,int,int'。"
        
        # 记录错误到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({"user_agent": user_agent}) if user_agent else None,
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': '400',
            'RES_ERROR_MESSAGE': 'Bad Request',
            'RES_DOCUMENT': None,
            'RES_MESSAGE': chinese_json_dumps({"error": error_msg}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'N',
            'ASR_VOICERESULT': None,
            'ASR_WAV_PATH': None
        }
        
        await save_to_database(db_record)
        raise HTTPException(status_code=400, detail=error_msg)

    # --- 读取音频文件内容 ---
    try:
        audio_bytes = await audio_file.read()
        wav_name = audio_file.filename if audio_file.filename else f"uploaded_audio_{req_seqno}"
        log.info(f"读取了 {len(audio_bytes)} 字节，来自 {wav_name}")
        
        if not audio_bytes:
            error_msg = "上传的音频文件为空。"
            
            # 记录错误到数据库
            db_record = {
                'TRAN_DATE': tran_date,
                'TRAN_TIMESTAMP': tran_timestamp,
                'REQ_SEQNO': req_seqno,
                'REQ_CHANNELID': req_channelid,
                'REQ_BANK_CODE': req_bank_code,
                'REQ_USERID': req_userid,
                'REQ_SERVICEID': req_serviceid,
                'REQ_MESSAGE': chinese_json_dumps({"user_agent": user_agent}) if user_agent else None,
                'REQ_DOCUMENT': req_document,
                'RES_ERROR_CODE': '400',
                'RES_ERROR_MESSAGE': 'Empty Audio',
                'RES_DOCUMENT': None,
                'RES_MESSAGE': chinese_json_dumps({"error": error_msg}),
                'SERVER_IP': server_ip,
                'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
                'ASR_FIND_WAV_PATH_STATUS': 'N',
                'ASR_VOICERESULT': None,
                'ASR_WAV_PATH': None
            }
            
            await save_to_database(db_record)
            raise HTTPException(status_code=400, detail=error_msg)
            
        # 如果需要，保存音频文件
        if save_wav:
            # 添加序列号到文件名以确保唯一性
            file_ext = os.path.splitext(wav_name)[1] if '.' in wav_name else '.wav'
            unique_filename = f"{os.path.splitext(wav_name)[0]}_{req_seqno}{file_ext}"
            wav_path = await save_audio_file(audio_bytes, unique_filename)
            
    except Exception as e:
        log.error(f"读取上传文件失败: {e}")
        error_msg = f"无法读取音频文件: {e}"
        
        # 记录错误到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({"user_agent": user_agent}) if user_agent else None,
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': '400',
            'RES_ERROR_MESSAGE': 'Read File Error',
            'RES_DOCUMENT': None,
            'RES_MESSAGE': chinese_json_dumps({"error": error_msg}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'N',
            'ASR_VOICERESULT': None,
            'ASR_WAV_PATH': None
        }
        
        await save_to_database(db_record)
        raise HTTPException(status_code=400, detail=error_msg)
    finally:
        await audio_file.close()  # 确保文件句柄被关闭

    # --- 调用WebSocket交互逻辑 ---
    try:
        transcription_result = await transcribe_audio_ws(
            host=host,
            port=port,
            chunk_size=chunk_size,
            chunk_interval=chunk_interval,
            hotword=hotword if hotword else "",  # 如果None则传递空字符串
            audio_bytes=audio_bytes,
            audio_fs=audio_fs,
            wav_name=wav_name,
            ssl_enabled=ssl_enabled,
            use_itn=use_itn,
            mode=mode,
            req_seqno=req_seqno,
            save_wav=save_wav
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        log.info(f"成功转录 {wav_name}，用时 {processing_time:.2f} 秒")
        
        # 记录成功结果到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({
                "mode": mode,
                "chunk_size": chunk_size_str,
                "chunk_interval": chunk_interval,
                "audio_fs": audio_fs,
                "use_itn": use_itn,
                "user_agent": user_agent,
                "hotword": hotword
            }),
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': '0000',
            'RES_ERROR_MESSAGE': '成功',
            'RES_DOCUMENT': None,
            'RES_MESSAGE': chinese_json_dumps({"responseCode": 200, "detail": "转录成功"}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{processing_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'Y' if wav_path else 'N',
            'ASR_VOICERESULT': chinese_json_dumps(transcription_result),
            'ASR_WAV_PATH': wav_path
        }
        
        await save_to_database(db_record)
        
        # 返回从FunASR接收的整个JSON消息
        return JSONResponse(content=transcription_result)

    except HTTPException as e:
        # 重新抛出HTTPExceptions，这些异常发生在WebSocket交互或验证过程中
        log.warning(f"转录 {wav_name} 失败，HTTPException: {e.status_code} - {e.detail}")
        
        # 记录错误到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({
                "mode": mode,
                "chunk_size": chunk_size_str,
                "chunk_interval": chunk_interval,
                "audio_fs": audio_fs,
                "use_itn": use_itn,
                "user_agent": user_agent,
                "hotword": hotword
            }),
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': str(e.status_code),
            'RES_ERROR_MESSAGE': 'HTTP异常',
            'RES_DOCUMENT': None,
            'RES_MESSAGE': chinese_json_dumps({"error": e.detail}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'Y' if wav_path else 'N',
            'ASR_VOICERESULT': None,
            'ASR_WAV_PATH': wav_path
        }
        
        await save_to_database(db_record)
        raise e
        
    except Exception as e:
        # 捕获处理过程中的任何其他未预期错误
        end_time = time.time()
        log.error(f"转录过程中出错，文件 {wav_name}，用时 {end_time - start_time:.2f} 秒: {e}")
        log.error(traceback.format_exc())
        
        # 记录错误到数据库
        db_record = {
            'TRAN_DATE': tran_date,
            'TRAN_TIMESTAMP': tran_timestamp,
            'REQ_SEQNO': req_seqno,
            'REQ_CHANNELID': req_channelid,
            'REQ_BANK_CODE': req_bank_code,
            'REQ_USERID': req_userid,
            'REQ_SERVICEID': req_serviceid,
            'REQ_MESSAGE': chinese_json_dumps({
                "mode": mode,
                "chunk_size": chunk_size_str,
                "chunk_interval": chunk_interval,
                "audio_fs": audio_fs,
                "use_itn": use_itn,
                "user_agent": user_agent,
                "hotword": hotword
            }),
            'REQ_DOCUMENT': req_document,
            'RES_ERROR_CODE': '500',
            'RES_ERROR_MESSAGE': '内部服务器错误',
            'RES_DOCUMENT': None,
            'RES_MESSAGE': chinese_json_dumps({"error": str(e)}),
            'SERVER_IP': server_ip,
            'TOTAL_TIME_USED': f"{time.time() - start_time:.3f}",
            'ASR_FIND_WAV_PATH_STATUS': 'Y' if wav_path else 'N',
            'ASR_VOICERESULT': None,
            'ASR_WAV_PATH': wav_path
        }
        
        await save_to_database(db_record)
        raise HTTPException(status_code=500, detail=f"转录期间内部服务器错误: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # 使用参数配置API服务器本身（可选）
    parser = argparse.ArgumentParser(description="运行FunASR HTTP API服务器")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="API服务器主机")
    parser.add_argument("--api_port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--workers", type=int, default=4, help="Uvicorn工作进程数（生产环境用）")
    parser.add_argument("--db_name", type=str, default="asr_data", help="数据库名称")
    parser.add_argument("--temp_dir", type=str, default=None, help="临时音频文件存储目录")
    api_args = parser.parse_args()
    
    # 更新数据库名称
    if api_args.db_name != "asr_data":
        DB_CONFIG['database'] = api_args.db_name
        print(f"使用数据库: {DB_CONFIG['database']}")
    
    # 更新临时目录
    if api_args.temp_dir:
        TEMP_AUDIO_DIR = api_args.temp_dir
        os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
        print(f"临时音频文件将保存到: {TEMP_AUDIO_DIR}")
    else:
        print(f"临时音频文件将保存到默认目录: {TEMP_AUDIO_DIR}")
    
    print(f"启动FunASR API服务器，地址: http://{api_args.api_host}:{api_args.api_port}")
    print(f"服务器IP: {get_server_ip()}")
    
    try:
        # 测试数据库连接
        conn = pymysql.connect(**DB_CONFIG)
        print("数据库连接测试成功！")
        conn.close()
    except Exception as e:
        print(f"警告：数据库连接测试失败: {e}")
        print("服务将启动，但数据库记录功能可能不可用。")

    # 使用"main:app"引用*此*文件中的app对象
    uvicorn.run(
        "main:app",  # 告诉uvicorn：在模块'main.py'中查找对象'app'
        host=api_args.api_host,
        port=api_args.api_port,
        workers=api_args.workers,  # 对于生产部署，如果合适，请考虑>1个工作进程
        # reload=True  # 仅在开发中启用reload，生产中禁用
                      # Uvicorn需要安装'standard' extras才能使用reload：pip install uvicorn[standard]
    )

"""
测试单声道识别：
curl -X POST "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&ssl=false&use_itn=true&audio_fs=16000&req_channelid=CURL&req_bank_code=TESTBANK&req_userid=TESTUSER" \
     -H "accept: application/json" \
     -F "audio_file=@../audio/asr_example.wav"

测试双声道识别：
curl -X POST "http://127.0.0.1:8000/transcribe_stereo?host=127.0.0.1&port=10098&mode=offline&ssl=false&use_itn=true&audio_fs=16000&req_channelid=CURL&req_bank_code=TESTBANK&req_userid=TESTUSER" \
     -H "accept: application/json" \
     -F "audio_file=@../audio/stereo_example.wav"
"""

"""
以上代码使用curosr中的Claude3.7 结合prompt文件中的详细提示生成，初步测试后功能正常，但是插入数据库中的RES_MESSAGE字段略微有问题，需要进一步优化。
"""