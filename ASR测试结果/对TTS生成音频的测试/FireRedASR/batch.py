import requests
import os
import time
import re # For extracting numbers from filenames

# --- Configuration ---
# !! IMPORTANT: Update these paths !!
AUDIO_DIR = "path/to/your/audio/folder"  # Directory containing sentence_1.wav, sentence_2.wav, etc.
OUTPUT_FILE = "transcription_results.txt" # File to save the results
API_URL = "http://localhost:5000/transcribe" # URL of your running Flask API
# Optional: Override API parameters via query string if needed
# API_PARAMS = {'beam_size': 5, 'use_gpu': 0} 
API_PARAMS = {} # Use API defaults by default

# --- Helper Function to Extract Number ---
def extract_number(filename):
    """Extracts the numerical part from filenames like 'sentence_123.wav'."""
    match = re.search(r'sentence_(\d+)\.wav$', filename)
    if match:
        return int(match.group(1))
    return None # Return None if the pattern doesn't match

# --- Main Processing Logic ---
def run_batch_transcription(audio_dir, output_file, api_url, api_params={}):
    """
    Finds audio files, sorts them, calls the transcription API, and saves results.
    """
    print(f"Starting batch transcription...")
    print(f"Input directory: {audio_dir}")
    print(f"Output file: {output_file}")
    print(f"API endpoint: {api_url}")

    if not os.path.isdir(audio_dir):
        print(f"Error: Input directory '{audio_dir}' not found.")
        return

    start_total_time = time.time()
    files_to_process = []

    # 1. Find and sort audio files
    print("Scanning directory for audio files...")
    try:
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith('.wav') and filename.lower().startswith('sentence_'):
                number = extract_number(filename)
                if number is not None:
                    full_path = os.path.join(audio_dir, filename)
                    files_to_process.append((number, filename, full_path))
                else:
                     print(f"Warning: Skipping file with unexpected name format: {filename}")
            
    except Exception as e:
        print(f"Error reading directory {audio_dir}: {e}")
        return

    if not files_to_process:
        print("No audio files matching 'sentence_*.wav' found in the directory.")
        return

    # Sort files based on the extracted number
    files_to_process.sort(key=lambda item: item[0])
    
    total_files = len(files_to_process)
    print(f"Found {total_files} files to process, sorted numerically.")

    # 2. Process files and write results
    results_count = 0
    error_count = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            print(f"Processing files and writing results to {output_file}...")
            for i, (number, filename, file_path) in enumerate(files_to_process):
                uttid = os.path.splitext(filename)[0] # Use filename without extension as uttid
                print(f"Processing file {i+1}/{total_files}: {filename} (uttid: {uttid})")
                
                try:
                    # Prepare file for upload
                    with open(file_path, 'rb') as audio_file:
                        files = {'audio_file': (filename, audio_file, 'audio/wav')}
                        data = {'uttid': uttid}

                        # Make the API request (using file upload method)
                        response = requests.post(api_url, files=files, data=data, params=api_params, timeout=300) # Increased timeout for potentially long ASR
                        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                        # Parse the result
                        result = response.json()

                        if 'error' in result:
                             print(f"  Error from API for {filename}: {result['error']}")
                             outfile.write(f"{filename}: [API Error: {result.get('details', result['error'])}]\n")
                             error_count += 1
                        elif 'text' in result:
                            transcription = result['text']
                            print(f"  Success: Transcription received.")
                            # Write result to file: "filename: transcription text"
                            outfile.write(f"{filename}: {transcription}\n")
                            results_count += 1
                        else:
                            print(f"  Error: Unexpected API response format for {filename}: {result}")
                            outfile.write(f"{filename}: [Error: Unexpected API response]\n")
                            error_count += 1

                except requests.exceptions.RequestException as e:
                    print(f"  Error: Network or API request failed for {filename}: {e}")
                    outfile.write(f"{filename}: [Error: Request failed - {e}]\n")
                    error_count += 1
                except Exception as e:
                    print(f"  Error: An unexpected error occurred processing {filename}: {e}")
                    outfile.write(f"{filename}: [Error: Processing failed - {e}]\n")
                    error_count += 1
                
                # Optional: Add a small delay between requests if needed
                # time.sleep(0.1) 

    except IOError as e:
        print(f"Error: Could not write to output file {output_file}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")
        return


    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    print("\n--- Batch Processing Summary ---")
    print(f"Total files processed: {results_count + error_count}/{total_files}")
    print(f"  Successful transcriptions: {results_count}")
    print(f"  Errors: {error_count}")
    print(f"Results saved to: {output_file}")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print("------------------------------")


# --- Run the Script ---
if __name__ == "__main__":
    # Make sure to update AUDIO_DIR and OUTPUT_FILE above before running!
    run_batch_transcription(AUDIO_DIR, OUTPUT_FILE, API_URL, API_PARAMS)