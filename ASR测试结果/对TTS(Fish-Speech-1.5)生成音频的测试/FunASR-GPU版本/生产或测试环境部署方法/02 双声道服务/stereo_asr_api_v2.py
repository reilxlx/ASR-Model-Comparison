# -*- encoding: utf-8 -*-
import os
import time
import asyncio
import json
import traceback
import argparse
import logging
import wave
import numpy as np
from io import BytesIO

# --- API Specific Imports ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import JSONResponse
import aiohttp # For async HTTP requests
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Stereo ASR API",
    description="Processes stereo audio by splitting channels, calling a mono ASR service for each, and merging results.",
)

# --- Helper Function to Call the Mono ASR API ---
async def call_mono_asr_api(
    session: aiohttp.ClientSession,
    mono_asr_url: str, # The full URL like http://127.0.0.1:8000/transcribe?host=...
    audio_bytes: bytes,
    channel_name: str # "A" or "B" for identification
) -> Optional[Dict[str, Any]]:
    """
    Sends a single mono audio stream to the underlying ASR API.

    Returns:
        The JSON response dictionary from the ASR API, or None if failed.
    """
    start_time = time.perf_counter()
    log.info(f"Sending request for channel {channel_name}...")

    form = aiohttp.FormData()
    form.add_field('audio_file',
                   audio_bytes,
                   filename=f"channel_{channel_name}.wav", # Filename for the form field
                   content_type='audio/wav')

    try:
        timeout = aiohttp.ClientTimeout(total=180) # 3 minutes timeout per channel

        async with session.post(mono_asr_url, data=form, timeout=timeout) as response:
            duration = time.perf_counter() - start_time
            response_text = await response.text()

            if response.status == 200:
                try:
                    result_json = json.loads(response_text)
                    log.info(f"Channel {channel_name} processed successfully in {duration:.2f}s.")
                    # Validation: Check for essential keys ('stamp_sents', 'text')
                    if ("stamp_sents" not in result_json or not isinstance(result_json["stamp_sents"], list) or
                        "text" not in result_json): # Added check for 'text' key
                         log.warning(f"Channel {channel_name} response lacks valid 'stamp_sents' or 'text'. Response: {result_json}")
                         return None
                    return result_json
                except json.JSONDecodeError:
                    log.error(f"Channel {channel_name}: Failed to decode JSON response. Status: {response.status}, Duration: {duration:.2f}s, Response: {response_text[:200]}...")
                    return None
            else:
                log.error(f"Channel {channel_name}: Received non-200 status: {response.status}. Duration: {duration:.2f}s, Response: {response_text[:200]}...")
                return None

    except asyncio.TimeoutError:
        duration = time.perf_counter() - start_time
        log.error(f"Channel {channel_name}: Request timed out after {duration:.2f}s.")
        return None
    except aiohttp.ClientError as e:
        duration = time.perf_counter() - start_time
        log.error(f"Channel {channel_name}: Client error during request after {duration:.2f}s: {e}")
        return None
    except Exception as e:
        duration = time.perf_counter() - start_time
        log.error(f"Channel {channel_name}: Unexpected error during request after {duration:.2f}s: {e}")
        log.error(traceback.format_exc())
        return None


# --- API Endpoint Definition ---
@app.post("/transcribe_stereo", response_class=JSONResponse)
async def http_transcribe_stereo(
    # --- Parameters for the underlying MONO ASR Service ---
    asr_host: str = Query("127.0.0.1", description="Host IP of the backend FunASR WebSocket server."),
    asr_port: int = Query(10098, description="Port of the backend FunASR WebSocket server."),
    asr_mode: str = Query("offline", description="Recognition mode for FunASR.", pattern="^(offline|online|2pass)$"),
    asr_ssl: bool = Query(False, description="Use wss:// for FunASR connection."),
    asr_use_itn: bool = Query(True, description="Apply ITN in FunASR."),
    asr_audio_fs: int = Query(16000, description="Sample rate expected by FunASR."),
    # --- Stereo Audio Input ---
    stereo_audio_file: UploadFile = File(..., description="Stereo WAV audio file to transcribe."),
):
    """
    Receives a stereo WAV file, splits channels, calls mono ASR API concurrently,
    merges results based on timestamps, and returns combined and individual transcriptions.
    """
    request_start_time = time.time()
    log.info(f"Received stereo transcription request for: {stereo_audio_file.filename}")

    # --- Read and Split Stereo Audio ---
    try:
        contents = await stereo_audio_file.read()
        await stereo_audio_file.close() # Close file handle

        with wave.open(BytesIO(contents), 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

            log.info(f"Audio Properties: Channels={n_channels}, Rate={frame_rate}, Width={sample_width}, Frames={n_frames}")

            if n_channels != 2:
                raise HTTPException(status_code=400, detail=f"Input file must be stereo (2 channels), got {n_channels}")
            if frame_rate != asr_audio_fs:
                log.warning(f"Input audio sample rate ({frame_rate} Hz) differs from target ASR rate ({asr_audio_fs} Hz). Using input rate for splitting, but ASR might perform suboptimally.")
            if sample_width != 2:
                 raise HTTPException(status_code=400, detail=f"Input file must be 16-bit (sample width 2), got {sample_width}")

            dtype = np.int16
            stereo_signal = np.frombuffer(audio_data, dtype=dtype)

            if stereo_signal.shape[0] == n_frames * n_channels:
                 stereo_signal = stereo_signal.reshape((-1, n_channels))
            else:
                 raise ValueError("Unexpected audio data shape after reading frames.")

            channel_a_signal = stereo_signal[:, 0].copy()
            channel_b_signal = stereo_signal[:, 1].copy()

            channel_a_bytes = channel_a_signal.tobytes()
            channel_b_bytes = channel_b_signal.tobytes()

            log.info(f"Successfully split stereo audio into Channel A ({len(channel_a_bytes)} bytes) and Channel B ({len(channel_b_bytes)} bytes).")

    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(f"Failed to read or split audio file: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {e}")

    # --- Construct URL for the underlying Mono ASR API ---
    mono_api_base_url = "http://127.0.0.1:8000" # IMPORTANT: Adjust if needed
    mono_api_path = "/transcribe"
    query_params = f"host={asr_host}&port={asr_port}&mode={asr_mode}&ssl={str(asr_ssl).lower()}&use_itn={str(asr_use_itn).lower()}&audio_fs={asr_audio_fs}"
    mono_asr_url = f"{mono_api_base_url}{mono_api_path}?{query_params}"
    log.info(f"Target Mono ASR API URL: {mono_asr_url}")

    # --- Make Concurrent Calls to Mono ASR API ---
    async with aiohttp.ClientSession() as session:
        task_a = asyncio.create_task(call_mono_asr_api(session, mono_asr_url, channel_a_bytes, "A"))
        task_b = asyncio.create_task(call_mono_asr_api(session, mono_asr_url, channel_b_bytes, "B"))

        results_a, results_b = await asyncio.gather(task_a, task_b)

    # --- Process and Merge Results ---
    if results_a is None or results_b is None:
        log.error("ASR failed for one or both channels.")
        raise HTTPException(status_code=502, detail="ASR processing failed for one or both audio channels. Check logs.")

    try:
        # --- Extract individual channel full texts ---
        full_text_a = results_a.get("text", "") # Get full text for channel A
        full_text_b = results_b.get("text", "") # Get full text for channel B

        # Extract segments and add speaker info
        segments_a = results_a.get("stamp_sents", [])
        segments_b = results_b.get("stamp_sents", [])

        merged_segments = []
        for seg in segments_a:
            seg['speaker'] = 'A'
            merged_segments.append(seg)
        for seg in segments_b:
            seg['speaker'] = 'B'
            merged_segments.append(seg)

        # Sort combined segments by start time
        merged_segments.sort(key=lambda x: x.get('start', 0))

        # Reconstruct merged full text based on sorted segments
        merged_full_text_parts = []
        for seg in merged_segments:
            text = seg.get('text_seg', '').replace(' ', '')
            punc = seg.get('punc', '')
            merged_full_text_parts.append(text + punc)

        merged_full_text = "".join(merged_full_text_parts)

        log.info("Successfully merged results from both channels.")

        # --- Construct Final Result ---
        final_result = {
            "status": "success",
            "original_filename": stereo_audio_file.filename,
            "full_text": merged_full_text,   # Merged text based on timestamp order
            "full_text_a": full_text_a,       # Full text from Channel A only
            "full_text_b": full_text_b,       # Full text from Channel B only
            "merged_segments": merged_segments, # Detailed segments with timestamps and speaker
        }
        processing_time = time.time() - request_start_time
        log.info(f"Stereo request completed in {processing_time:.2f} seconds.")
        return JSONResponse(content=final_result)

    except Exception as e:
        log.error(f"Failed to merge or format results: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error merging ASR results: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stereo ASR API Server")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="Host for this Stereo API server")
    parser.add_argument("--api_port", type=int, default=8001, help="Port for this Stereo API server")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers")
    api_args = parser.parse_args()

    print(f"Starting Stereo ASR API Server on http://{api_args.api_host}:{api_args.api_port}")
    print("Ensure the underlying Mono ASR API server is running and accessible.")

    uvicorn.run(
        "stereo_asr_api:app",
        host=api_args.api_host,
        port=api_args.api_port,
        workers=api_args.workers,
    )


# curl -X POST "http://127.0.0.1:8001/transcribe_stereo?asr_host=127.0.0.1&asr_port=10098&asr_mode=offline&asr_ssl=false&asr_use_itn=true&asr_audio_fs=16000" \
#      -H "accept: application/json" \
#      -F "stereo_audio_file=@./stereo_example.wav" # Path to your STEREO audio file