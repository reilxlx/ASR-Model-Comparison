import requests
import os
import argparse
import re
import logging
import time
from datetime import timedelta # To format time nicely

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_number(filename):
    """Extracts the numerical part from filenames like 'sentence_1.wav'."""
    match = re.search(r'(\d+)\.wav$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None # Return None if no number found or pattern doesn't match

def transcribe_folder(api_url, input_dir, output_file):
    """
    Finds .wav files in input_dir, calls the transcription API for each in
    numerical order, writes results to output_file, and calculates timing.
    """
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    files_to_process = []
    logging.info(f"Scanning directory: {input_dir}")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            number = extract_number(filename)
            if number is not None:
                full_path = os.path.join(input_dir, filename)
                files_to_process.append((number, full_path, filename)) # Store number, path, and original name
            else:
                logging.warning(f"Skipping file (could not extract number or wrong format): {filename}")
        else:
            logging.debug(f"Skipping non-wav file: {filename}")

    # Sort files based on the extracted number
    files_to_process.sort(key=lambda item: item[0])

    if not files_to_process:
        logging.warning(f"No WAV files matching the pattern 'name_NUMBER.wav' found in {input_dir}")
        return

    total_files = len(files_to_process)
    logging.info(f"Found {total_files} files to process. Writing results to: {output_file}")

    # --- Timing Initialization ---
    batch_start_time = time.time()
    successful_files = 0
    failed_files = 0
    total_processing_time_successful = 0.0 # Sum of durations for successful requests

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, (number, file_path, filename) in enumerate(files_to_process):
                logging.info(f"Processing file {i+1}/{total_files}: {filename} (Number: {number})")
                file_processed_successfully = False
                request_duration = 0.0
                try:
                    with open(file_path, 'rb') as audio_file_handle:
                        # Prepare the file for upload
                        files = {'audio_file': (filename, audio_file_handle, 'audio/wav')}
                        headers = {'accept': 'application/json'}

                        request_start_time = time.time()
                        # Make the POST request
                        response = requests.post(api_url, headers=headers, files=files, timeout=300) # Increased timeout
                        request_end_time = time.time()
                        request_duration = request_end_time - request_start_time

                        # Check response status
                        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                        # Parse the JSON response
                        result_json = response.json()

                        # Extract the transcription text
                        transcription_text = result_json.get('text')

                        if transcription_text is not None:
                            logging.info(f"  Success ({request_duration:.2f}s): {transcription_text[:100]}...") # Log snippet
                            # Write the original filename and the transcription to the output file
                            outfile.write(f"{filename}\t{transcription_text}\n")
                            file_processed_successfully = True
                        else:
                            logging.warning(f"  'text' key not found in response for {filename}. Response: {result_json}")
                            outfile.write(f"{filename}\tERROR: 'text' key missing in response\n")

                except requests.exceptions.RequestException as e:
                    logging.error(f"  Error processing {filename}: Request failed: {e}")
                    outfile.write(f"{filename}\tERROR: Request failed - {e}\n")
                except requests.exceptions.HTTPError as e:
                    logging.error(f"  Error processing {filename}: HTTP Error: {e.response.status_code} - {e.response.text[:200]}...") # Log part of error response
                    outfile.write(f"{filename}\tERROR: HTTP {e.response.status_code}\n")
                except Exception as e:
                    logging.error(f"  An unexpected error occurred while processing {filename}: {e}")
                    outfile.write(f"{filename}\tERROR: Unexpected error - {e}\n")
                finally:
                    # Update counters based on success/failure
                    if file_processed_successfully:
                        successful_files += 1
                        total_processing_time_successful += request_duration # Add duration only for successful ones
                    else:
                        failed_files += 1


        logging.info("Finished processing all files.")

    except IOError as e:
        logging.error(f"Could not open output file {output_file} for writing: {e}")
        return # Stop if output can't be opened
    except Exception as e:
        logging.error(f"An unexpected error occurred during file writing: {e}")
        # Continue to print summary even if there was a writing error mid-way? Or return? Let's continue for now.

    # --- Timing Calculation and Output ---
    batch_end_time = time.time()
    total_batch_duration = batch_end_time - batch_start_time
    average_time_per_successful_file = (total_processing_time_successful / successful_files) if successful_files > 0 else 0.0

    logging.info("-" * 30)
    logging.info("Batch Processing Summary")
    logging.info("-" * 30)
    logging.info(f"Total files found:         {total_files}")
    logging.info(f"Successfully processed:    {successful_files}")
    logging.info(f"Failed to process:         {failed_files}")
    logging.info(f"Total batch duration:      {timedelta(seconds=total_batch_duration)}")
    if successful_files > 0:
         logging.info(f"Total API request time (successful): {timedelta(seconds=total_processing_time_successful)}")
         logging.info(f"Average API request time per successful file: {average_time_per_successful_file:.2f} seconds")
    logging.info("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch transcribe WAV files in a folder using FunASR API.")
    parser.add_argument("--input-dir", required=True, help="Directory containing the WAV audio files (e.g., sentence_1.wav).")
    parser.add_argument("--output-file", required=True, help="Path to the output text file to store transcriptions.")
    parser.add_argument("--api-url", required=True, help="Full URL of the FunASR transcription API endpoint (including query parameters like mode, host, port etc.). Example: 'http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&ssl=false&use_itn=true&audio_fs=16000'")

    args = parser.parse_args()

    transcribe_folder(args.api_url, args.input_dir, args.output_file)