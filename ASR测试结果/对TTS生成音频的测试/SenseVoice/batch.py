import os
import requests
import json
import time # Import the time module
from natsort import natsorted # For natural sorting (e.g., sentence_1, sentence_2, ..., sentence_10)

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/transcribe"  # Your running API endpoint
AUDIO_FOLDER = "./audio_files"              # Folder containing sentence_*.wav files
OUTPUT_TXT_FILE = "transcription_results.txt" # File to save results
ALLOWED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.pcm', '.m4a') # Allowed audio extensions
FILENAME_PREFIX = "sentence_" # Optional: Process only files starting with this prefix
REQUEST_TIMEOUT = 120 # Timeout for each API request in seconds
# --- End Configuration ---

def transcribe_single_file(audio_path):
    """Calls the transcription API for a single audio file."""
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {'audio_file': (os.path.basename(audio_path), audio_file)}
            response = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            transcription = result.get('transcription')
            if transcription is not None:
                return transcription
            else:
                print(f"Warning: API response for {os.path.basename(audio_path)} did not contain 'transcription' key. Response: {result}")
                return None
    except requests.exceptions.Timeout:
        print(f"Error: API request timed out for {os.path.basename(audio_path)} after {REQUEST_TIMEOUT} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling API for {os.path.basename(audio_path)}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {os.path.basename(audio_path)}. Status: {response.status_code}. Response text: {response.text[:200]}...")
        return None
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {os.path.basename(audio_path)}: {e}")
        return None

def process_audio_folder(folder_path, output_file):
    """Finds audio files, calls API for each, saves results, and calculates total time."""
    if not os.path.isdir(folder_path):
        print(f"Error: Input folder '{folder_path}' not found or is not a directory.")
        return

    audio_files_to_process = []
    try:
        all_files = os.listdir(folder_path)
        for filename in all_files:
            if (FILENAME_PREFIX is None or filename.startswith(FILENAME_PREFIX)) and \
               filename.lower().endswith(ALLOWED_EXTENSIONS):
                full_path = os.path.join(folder_path, filename)
                if os.path.isfile(full_path):
                    audio_files_to_process.append(full_path)
    except OSError as e:
        print(f"Error listing files in directory '{folder_path}': {e}")
        return

    if not audio_files_to_process:
        print(f"No audio files matching the criteria found in '{folder_path}'.")
        return

    sorted_audio_files = natsorted(audio_files_to_process)
    file_count = len(sorted_audio_files)
    print(f"Found {file_count} audio files to process. Order:")
    for i, f in enumerate(sorted_audio_files):
        print(f"  {i+1}. {os.path.basename(f)}")

    print(f"\nStarting transcription process. Results will be saved to '{output_file}'...")

    # --- Timing Start ---
    start_time = time.perf_counter()
    # --- ---

    success_count = 0
    fail_count = 0

    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, audio_path in enumerate(sorted_audio_files):
                filename = os.path.basename(audio_path)
                print(f"Processing [{i+1}/{file_count}] '{filename}'...")
                transcription = transcribe_single_file(audio_path)

                if transcription is not None:
                    f_out.write(f"{filename}: {transcription}\n")
                    print(f"  -> Success.")
                    success_count += 1
                else:
                    f_out.write(f"{filename}: [ERROR - Transcription Failed or Missing]\n")
                    print(f"  -> Failed.")
                    fail_count += 1

        # --- Timing End ---
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        # --- ---

        print(f"\n----- Processing Summary -----")
        print(f"Results saved to: '{output_file}'")
        print(f"Total files processed: {file_count}")
        print(f"Successful transcriptions: {success_count}")
        print(f"Failed transcriptions: {fail_count}")
        print(f"Total processing time: {total_duration:.2f} seconds") # Output total time
        if file_count > 0:
             avg_time = total_duration / file_count
             print(f"Average time per file: {avg_time:.2f} seconds")
        print(f"-----------------------------")


    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")
        # Optionally report partial time if needed, but usually erroring out is enough
        # end_time = time.perf_counter()
        # total_duration = end_time - start_time
        # print(f"Processing stopped due to error after {total_duration:.2f} seconds.")


# --- Main Execution ---
if __name__ == "__main__":
    process_audio_folder(AUDIO_FOLDER, OUTPUT_TXT_FILE)