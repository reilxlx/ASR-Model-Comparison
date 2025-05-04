import requests
import os
import re
import sys
from tqdm import tqdm # Optional: for progress bar

# --- Configuration ---
# URL of your running Flask API endpoint
API_URL = "http://localhost:5001/transcribe"
# Directory containing the audio files (sentence_1.wav, sentence_2.wav, ...)
AUDIO_DIR = "/path/to/your/audio/folder" # <<<--- CHANGE THIS TO YOUR AUDIO FOLDER PATH
# Output file to save the transcriptions
OUTPUT_FILE = "transcription_results.txt"
# Timeout for API requests in seconds
REQUEST_TIMEOUT = 60 # Increase if your audio files are very long

# --- Helper function for numerical sorting ---
def numerical_sort_key(filename):
    """Extracts the number from filenames like 'sentence_X.wav' for sorting."""
    match = re.search(r'sentence_(\d+)\.wav$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return float('inf') # Put files that don't match the pattern at the end

# --- Main Script ---
def main():
    # Validate audio directory
    if not os.path.isdir(AUDIO_DIR):
        print(f"Error: Audio directory not found: {AUDIO_DIR}")
        sys.exit(1)

    print(f"Looking for audio files in: {AUDIO_DIR}")

    # Get all .wav files matching the pattern
    try:
        all_files = os.listdir(AUDIO_DIR)
        audio_files = [
            f for f in all_files
            if f.lower().startswith('sentence_') and f.lower().endswith('.wav')
               and os.path.isfile(os.path.join(AUDIO_DIR, f))
        ]
    except OSError as e:
        print(f"Error reading directory {AUDIO_DIR}: {e}")
        sys.exit(1)


    if not audio_files:
        print(f"No files matching 'sentence_*.wav' found in {AUDIO_DIR}.")
        sys.exit(0)

    # Sort files numerically based on the number in the filename
    audio_files.sort(key=numerical_sort_key)

    print(f"Found {len(audio_files)} audio files to process. Sorted list:")
    # Print first few and last few for verification if list is long
    if len(audio_files) > 10:
        print("  ", audio_files[:5], "...")
        print("  ", audio_files[-5:])
    else:
        print("  ", audio_files)

    print(f"Results will be saved to: {OUTPUT_FILE}")

    # Open the output file for writing (use UTF-8 for Chinese characters)
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            # Use tqdm for progress bar if installed
            file_iterator = tqdm(audio_files, desc="Transcribing") if 'tqdm' in sys.modules else audio_files

            for filename in file_iterator:
                file_path = os.path.join(AUDIO_DIR, filename)
                transcription = None
                error_message = None

                try:
                    # Open the audio file in binary read mode
                    with open(file_path, 'rb') as audio_data:
                        # Prepare the files payload for the POST request
                        files = {'audio_file': (filename, audio_data, 'audio/wav')}

                        # Send the request to the API
                        response = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)

                        # Check if the request was successful
                        if response.status_code == 200:
                            try:
                                result_json = response.json()
                                if 'transcription' in result_json:
                                    transcription = result_json['transcription']
                                else:
                                    error_message = f"API Error: 'transcription' key missing in response. Raw: {response.text[:100]}..." # Limit raw output length
                            except requests.exceptions.JSONDecodeError:
                                error_message = f"API Error: Invalid JSON received. Raw: {response.text[:100]}..."
                        else:
                            error_message = f"API Error: Status Code {response.status_code}. Response: {response.text[:100]}..."

                except requests.exceptions.Timeout:
                    error_message = f"Error: Request timed out after {REQUEST_TIMEOUT} seconds."
                except requests.exceptions.RequestException as e:
                    error_message = f"Error: Network request failed: {e}"
                except FileNotFoundError:
                    error_message = f"Error: File not found (should not happen if listed): {file_path}"
                except Exception as e:
                    error_message = f"Error: An unexpected error occurred: {e}"

                # Write result or error to the output file
                if transcription is not None:
                    outfile.write(transcription + '\n')
                else:
                    # Log error to console and write an indicator to the file
                    error_log = f"FAILED: {filename} - {error_message}"
                    print(error_log, file=sys.stderr) # Print errors to stderr
                    outfile.write(f"ERROR_PROCESSING_FILE: {filename} - {error_message}\n")

    except IOError as e:
        print(f"Error: Could not open or write to output file {OUTPUT_FILE}: {e}")
        sys.exit(1)

    print(f"\nBatch transcription complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()