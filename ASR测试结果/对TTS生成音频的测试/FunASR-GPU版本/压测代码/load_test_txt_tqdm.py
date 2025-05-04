import asyncio
import aiohttp
import time
import argparse
import logging
import json
import os
from collections import defaultdict
from datetime import timedelta

# --- Add tqdm imports ---
from tqdm.asyncio import tqdm as asyncio_tqdm # For async progress bars
from tqdm import tqdm as standard_tqdm      # For regular loops

# Configure logging (adjust level if tqdm output interferes, e.g., logging.WARNING)
# Tqdm writes to stderr, logging usually to stdout or stderr. Might need tweaking if output is messy.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Define the EXACT expected transcription text ---
EXPECTED_TEXT = "我们注意到您近期咨询了关于我行星推出的一款结构性理财产品。这款产品本金相对安全，其预期收益与特定的市场指标，例如某个股票指数或汇率挂钩，请您知悉。虽然它提供了潜在的较高回报机会，但实际收益可能存在不确定性，极端情况下可能仅能获得较低的保底收益甚至零收益。请在投资前确认，以充分理解产品结构和风险。"


# --- Core Request Function (Unchanged) ---
async def make_request(session: aiohttp.ClientSession, url: str, audio_bytes: bytes, audio_filename: str):
    """
    Sends a single POST request to the API endpoint and measures performance.

    Returns:
        tuple: (is_success, duration, status_code, error_message or None)
               is_success is True ONLY if status is 200 and 'text' field value
               exactly matches EXPECTED_TEXT.
    """
    start_time = time.perf_counter()
    error_message = None
    status_code = None
    is_success = False

    form = aiohttp.FormData()
    form.add_field('audio_file',
                   audio_bytes,
                   filename=audio_filename,
                   content_type='audio/wav')

    try:
        timeout = aiohttp.ClientTimeout(total=120) # 2 minutes total timeout

        async with session.post(url, data=form, timeout=timeout) as response:
            status_code = response.status
            response_text = await response.text() # Read text first for logging errors

            if response.status == 200:
                try:
                    result_json = json.loads(response_text)
                    actual_text = result_json.get("text")

                    # --- MODIFIED SUCCESS CHECK ---
                    if actual_text == EXPECTED_TEXT:
                        is_success = True
                    elif actual_text is not None: # Text exists but doesn't match
                        is_success = False
                        error_message = "Text mismatch"
                        log.debug(f"Text mismatch. Expected(start): '{EXPECTED_TEXT[:30]}...', Got(start): '{actual_text[:30]}...'")
                    else: # Text field is missing or None
                        is_success = False
                        error_message = "Response JSON lacks 'text' field"
                        log.debug(f"Missing 'text' in response: {result_json}")
                    # --- END MODIFIED SUCCESS CHECK ---

                except json.JSONDecodeError:
                    is_success = False
                    error_message = "Failed to decode JSON response"
                    log.warning(f"JSON Decode Error. Status: {status_code}, Response: {response_text[:200]}...")
            else:
                is_success = False
                error_message = f"HTTP Error {status_code}"
                log.warning(f"HTTP Error. Status: {status_code}, Response: {response_text[:200]}...")

    except aiohttp.ClientConnectorError as e:
        error_message = f"Connection Error: {e}"
        # Avoid logging error here if tqdm is active, keep it for summary
    except asyncio.TimeoutError:
        error_message = "Request Timeout"
    except aiohttp.ClientError as e: # Catch other aiohttp client errors
        error_message = f"Client Error: {e}"
    except Exception as e:
        error_message = f"Unexpected Error: {e}"
        # Maybe log exception here if it's truly unexpected
        # log.exception("Unexpected error during request")
    finally:
         if error_message is not None:
             is_success = False


    end_time = time.perf_counter()
    duration = end_time - start_time
    # Return error message for detailed analysis
    return is_success, duration, status_code, error_message


# --- Test Runner Function (Modified for Progress Bar) ---
async def run_test(url: str, audio_file_path: str, concurrency: int, num_requests: int):
    """
    Runs a load test with specified concurrency and number of requests.
    Uses the modified success criteria and displays a progress bar.
    """
    # Log initial info before the progress bar starts
    log.info(f"Success Criterion: Exact match for text starting with '{EXPECTED_TEXT[:30]}...'")

    if not os.path.exists(audio_file_path):
        log.error(f"Audio file not found: {audio_file_path}")
        return None

    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
        audio_filename = os.path.basename(audio_file_path)
        log.info(f"Loaded audio file: {audio_filename} ({len(audio_bytes)} bytes)")
    except Exception as e:
        log.error(f"Failed to read audio file {audio_file_path}: {e}")
        return None

    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    test_start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:

        async def guarded_request(req_num):
            async with semaphore:
                # Removed logging inside the guarded request to avoid flooding with tqdm active
                # log.debug(f"Req {req_num+1}: Starting request...")
                result = await make_request(session, url, audio_bytes, audio_filename)
                # log.debug(f"Req {req_num+1}: Finished. Success={result[0]}, Duration={result[1]:.3f}s, Err='{result[3]}'")
                return result

        # Create all tasks first
        for i in range(num_requests):
            tasks.append(asyncio.create_task(guarded_request(i)))

        # --- Use tqdm.asyncio.gather for progress bar ---
        # Leave setting ensures the bar stays visible after completion
        # Unit clarifies what the bar is counting
        results = await asyncio_tqdm.gather(
            *tasks,
            total=num_requests,
            desc=f"Conc={concurrency:<3}", # Description for the progress bar
            unit="req",
            leave=True # Keep the bar after finishing
        )
        # --- End progress bar usage ---

    test_end_time = time.perf_counter()
    total_test_duration = test_end_time - test_start_time

    # --- Analyze Results (Unchanged logic, but logging reduced above) ---
    successful_requests = 0
    failed_requests = 0
    total_duration_successful = 0.0
    status_code_counts = defaultdict(int)
    error_counts = defaultdict(int)

    for result in results:
        if isinstance(result, Exception):
            failed_requests += 1
            error_message = f"Task Exception: {type(result).__name__}"
            error_counts[error_message] += 1
            # Log task exceptions as they are less common and important
            log.error(f"Caught exception in task result: {result}")
        elif isinstance(result, tuple) and len(result) == 4:
            is_success, duration, status_code, error_msg = result
            if status_code:
                 status_code_counts[status_code] += 1

            if is_success:
                successful_requests += 1
                total_duration_successful += duration
            else:
                failed_requests += 1
                error_key = error_msg if error_msg else "Unknown Failure"
                error_counts[error_key] += 1
        else:
             failed_requests += 1
             error_counts["Invalid Result Format"] += 1
             log.error(f"Received unexpected result format: {result}")

    success_rate = (successful_requests / num_requests * 100) if num_requests > 0 else 0
    avg_duration_successful = (total_duration_successful / successful_requests) if successful_requests > 0 else 0.0

    # --- Print Summary (Log after progress bar is done) ---
    # Use print or ensure logger is configured properly if tqdm interferes
    print(f"\n--- Test Summary: Concurrency={concurrency} ---")
    print(f"Total Requests Sent:      {num_requests}")
    print(f"Successful Requests (Exact Match): {successful_requests}")
    print(f"Failed Requests:          {failed_requests}")
    print(f"Success Rate (Exact Match): {success_rate:.2f}%")
    print(f"Total Test Duration:      {timedelta(seconds=total_test_duration)}")
    if successful_requests > 0:
        print(f"Avg. Resp. Time (Success):{avg_duration_successful:.3f} seconds")
    else:
        print(f"Avg. Resp. Time (Success): N/A (0 successful requests)")
    print(f"Status Code Counts:       {dict(status_code_counts)}")
    if error_counts:
        print(f"Failure Reason Counts:    {dict(error_counts)}")
    print(f"--- Test Finished: Concurrency={concurrency} ---\n")


    # Return stats for potential overall summary
    return {
        "concurrency": concurrency,
        "total_requests": num_requests,
        "successful": successful_requests,
        "failed": failed_requests,
        "success_rate": success_rate,
        "total_duration": total_test_duration,
        "avg_duration_successful": avg_duration_successful,
        "status_codes": dict(status_code_counts),
        "errors": dict(error_counts)
    }

# --- Main Execution (Modified for Overall Progress Bar) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the FunASR API endpoint with exact text matching and progress bars.")
    parser.add_argument("--url", required=True, help="Full URL of the FunASR transcription API endpoint (including query parameters).")
    parser.add_argument("--audio-file", required=True, help="Path to the specific WAV audio file expected to produce the reference text.")
    parser.add_argument("--concurrency-levels", default="1,5,10", help="Comma-separated list of concurrency levels to test (e.g., '1,5,10').")
    parser.add_argument("--requests-per-level", type=int, default=50, help="Number of requests to send for each concurrency level.")

    args = parser.parse_args()

    try:
        concurrency_levels = [int(c.strip()) for c in args.concurrency_levels.split(',')]
    except ValueError:
        log.error("Invalid format for --concurrency-levels. Use comma-separated integers (e.g., '1,5,10').")
        exit(1)

    if args.requests_per_level <= 0:
         log.error("--requests-per-level must be positive.")
         exit(1)

    if not os.path.exists(args.audio_file):
        log.error(f"Specified audio file not found: {args.audio_file}")
        exit(1)

    # Print initial info before any progress bars start
    print("Starting load test...")
    print(f"API URL: {args.url}")
    print(f"Audio File: {args.audio_file}")
    print(f"Requests per level: {args.requests_per_level}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Expected Text (start): '{EXPECTED_TEXT[:50]}...'")
    print("-" * 40)


    all_results = []
    # --- Wrap the main loop with standard tqdm for overall progress ---
    # The description shows which concurrency level is next/running
    # `position=0` tries to keep this bar at the top
    # `leave=True` keeps the overall bar visible after completion
    overall_progress_bar = standard_tqdm(
        concurrency_levels,
        desc="Overall Test Progress",
        position=0,
        leave=True,
        unit="level"
    )

    for conc_level in overall_progress_bar:
        # Update overall bar description
        overall_progress_bar.set_description(f"Running Conc={conc_level}")

        # --- Run the specific concurrency level test ---
        # Logs inside run_test handle the start/loaded messages
        # The asyncio_tqdm bar will appear below the overall bar
        stats = asyncio.run(run_test(args.url, args.audio_file, conc_level, args.requests_per_level))

        if stats:
             all_results.append(stats)
        # Optionally add a small delay if output gets jumbled
        # time.sleep(0.1)

    # Ensure overall bar shows completion
    overall_progress_bar.set_description("Overall Test Complete")
    overall_progress_bar.close() # Close the overall bar


    print("="*40)
    print("Overall Load Test Complete")
    print("="*40)
    # You can still print the collected results if needed
    # print("Collected Results:")
    # for res in all_results:
    #     print(res)

# python load_test_by_txt_tqdm.py \
#     --url "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10096&mode=offline&ssl=false&use_itn=true&audio_fs=16000" \
#     --audio-file "/root/home/fish-speech-liutao-16k/sentence_104.wav" \
#     --concurrency-levels "1,5,10,15,20,25,30" \
#     --requests-per-level 500