import os
import glob
import soundfile as sf # Keep import, though path is used directly by KimiAudio
from kimia_infer.api.kimia import KimiAudio
import torch
import re # For natural sorting
import time # Import the time module

# --- Configuration ---
# *** IMPORTANT: UPDATE THESE PATHS ***
AUDIO_DIR = "/root/home/fish-speech-liutao/" # Directory containing sentence_1.wav, sentence_2.wav, etc.
OUTPUT_TXT_PATH = "batch_transcription_results.txt" # File to save the results
MODEL_ID = "/root/.cache/modelscope/hub/models/moonshotai/Kimi-Audio-7B-Instruct" # Or your specific model path/ID
# Fallback local path example (uncomment and update if needed)
# MODEL_ID = "/path/to/your/downloaded/kimia-hf-ckpt"

# --- 1. Load Model ---
print("Loading Kimi-Audio model...")
load_start_time = time.perf_counter() # Optional: time the loading itself if interested
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = None # Initialize model to None
try:
    model = KimiAudio(model_path=MODEL_ID, load_detokenizer=True)
    # model.to(device)
    load_end_time = time.perf_counter() # Optional: end loading timer
    print(f"Model loaded successfully in {load_end_time - load_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading model from {MODEL_ID}: {e}")
    print("Please ensure the model path is correct and dependencies are installed.")
    exit(1) # Exit if model loading fails

# --- Start Processing Timer (AFTER model load) ---
processing_start_time = time.perf_counter()

# --- 2. Define Sampling Parameters ---
sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

# --- 3. Batch Processing ---
print(f"\nStarting batch transcription for files in: {AUDIO_DIR}")
print(f"Results will be saved to: {OUTPUT_TXT_PATH}")

# Find all .wav files in the directory
audio_files = glob.glob(os.path.join(AUDIO_DIR, "sentence_*.wav"))

# Function to extract number for sorting
def extract_number(filename):
    basename = os.path.basename(filename)
    match = re.search(r'sentence_(\d+)\.wav', basename)
    return int(match.group(1)) if match else float('inf') # Put non-matching names last

# Sort files naturally based on the number
sorted_audio_files = sorted(audio_files, key=extract_number)

if not sorted_audio_files:
    print(f"Error: No files matching 'sentence_*.wav' found in {AUDIO_DIR}")
    exit(1)

print(f"Found {len(sorted_audio_files)} audio files to process.")

total_files = len(sorted_audio_files)
processed_files = 0
error_files = 0

# Open the output file
try:
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as outfile:
        for i, audio_path in enumerate(sorted_audio_files):
            filename = os.path.basename(audio_path)
            print(f"\nProcessing file {i+1}/{total_files}: {filename}...")

            # Check if file exists (though glob should ensure this)
            if not os.path.exists(audio_path):
                print(f"  Warning: File not found {audio_path}. Skipping.")
                outfile.write(f"{filename}: [Error: File not found]\n")
                error_files += 1
                continue

            # Construct messages for the model
            messages_asr = [
                {"role": "user", "message_type": "text", "content": "Please transcribe the following audio accurately:"},
                {"role": "user", "message_type": "audio", "content": audio_path}
            ]

            try:
                # Generate only text output
                _, text_output = model.generate(messages_asr, **sampling_params, output_type="text")

                print(f"  Transcription: {text_output}")
                # Write result to file
                outfile.write(f"{text_output}\n") # Only writing transcription per user request
                # outfile.write(f"{filename}: {text_output}\n") # Alternative: write filename too
                processed_files += 1

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                # Write error message to the output file for tracking
                outfile.write(f"{filename}: [Error: {e}]\n")
                error_files += 1

except IOError as e:
    print(f"Error opening or writing to output file {OUTPUT_TXT_PATH}: {e}")
    # --- Stop processing timer even if there's an I/O error before finishing ---
    processing_end_time = time.perf_counter()
    elapsed_processing_time = processing_end_time - processing_start_time
    print(f"\n--- Processing Summary ---")
    print(f"Attempted to process: {total_files} files")
    print(f"Successfully processed: {processed_files} files")
    print(f"Errors encountered: {error_files} files")
    print(f"Total processing time (excluding model loading): {elapsed_processing_time:.2f} seconds")
    print(f"\nBatch transcription INTERRUPTED due to file error. Partial results saved to {OUTPUT_TXT_PATH}")
except Exception as e:
    print(f"An unexpected error occurred during batch processing: {e}")
    # --- Stop processing timer in case of other unexpected errors ---
    processing_end_time = time.perf_counter()
    elapsed_processing_time = processing_end_time - processing_start_time
    print(f"\n--- Processing Summary ---")
    print(f"Attempted to process: {total_files} files")
    print(f"Successfully processed: {processed_files} files")
    print(f"Errors encountered: {error_files} files")
    print(f"Total processing time (excluding model loading): {elapsed_processing_time:.2f} seconds")
    print(f"\nBatch transcription FAILED due to unexpected error. Partial results may be in {OUTPUT_TXT_PATH}")
finally:
    # --- Stop Processing Timer (AFTER loop finishes or error occurs) ---
    # This block executes even if errors occurred in the try block (unless it was an exit())
    # We check if the end time was already set due to an error handled in except blocks
    if 'processing_end_time' not in locals():
         processing_end_time = time.perf_counter()
         elapsed_processing_time = processing_end_time - processing_start_time

         print(f"\n--- Processing Summary ---")
         print(f"Attempted to process: {total_files} files")
         print(f"Successfully processed: {processed_files} files")
         print(f"Errors encountered: {error_files} files")
         print(f"Total processing time (excluding model loading): {elapsed_processing_time:.2f} seconds")
         print(f"\nBatch transcription finished. Results saved to {OUTPUT_TXT_PATH}")