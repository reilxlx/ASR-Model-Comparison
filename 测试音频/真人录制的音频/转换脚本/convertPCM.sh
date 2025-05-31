#!/bin/bash

# --- Configuration ---
FFMPEG_OPTS_PCM_16K_S16LE_MONO="-f s16le -ar 16000 -ac 1 -acodec pcm_s16le"

# --- Helper Functions ---
usage() {
  echo "Usage: $0 <input_wav_directory> <output_pcm_directory>"
  echo "Example: $0 ./my_wavs ./my_pcms"
  exit 1
}

log_info() {
  echo "[INFO] $1"
}

log_error() {
  echo "[ERROR] $1" >&2
}

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
  usage
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
  log_error "ffmpeg could not be found. Please install ffmpeg."
  exit 1
fi

# Check if input directory exists and is a directory
if [ ! -d "$INPUT_DIR" ]; then
  log_error "Input directory '$INPUT_DIR' not found or is not a directory."
  exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
  log_info "Output directory '$OUTPUT_DIR' not found. Creating it..."
  mkdir -p "$OUTPUT_DIR"
  if [ $? -ne 0 ]; then
    log_error "Could not create output directory '$OUTPUT_DIR'."
    exit 1
  fi
else
  log_info "Output directory '$OUTPUT_DIR' already exists."
fi

# Ensure output directory is writable
if [ ! -w "$OUTPUT_DIR" ]; then
    log_error "Output directory '$OUTPUT_DIR' is not writable."
    exit 1
fi

# --- Main Processing Logic ---
log_info "Starting WAV to PCM conversion..."
log_info "Input directory: $INPUT_DIR"
log_info "Output directory: $OUTPUT_DIR"
log_info "FFmpeg options: $FFMPEG_OPTS_PCM_16K_S16LE_MONO"
echo "----------------------------------------"

converted_count=0
skipped_count=0
error_count=0

# Find .wav files (case-insensitive) in the input directory (non-recursively)
# -print0 and read -d $'\0' handle filenames with spaces or special characters
find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.wav" -print0 | while IFS= read -r -d $'\0' wav_file; do
  filename_with_ext=$(basename "$wav_file")
  filename_no_ext="${filename_with_ext%.*}" # Removes the last extension (e.g., .wav or .WAV)
  output_pcm_file="$OUTPUT_DIR/${filename_no_ext}.pcm"

  log_info "Processing: '$wav_file'"

  # ffmpeg command
  # -y: Overwrite output files without asking
  # -v error: Show only errors. Use -v quiet -stats for progress. Use -v verbose for more details.
  if ffmpeg -i "$wav_file" $FFMPEG_OPTS_PCM_16K_S16LE_MONO "$output_pcm_file" -y -v error; then
    log_info "Successfully converted to: '$output_pcm_file'"
    ((converted_count++))
  else
    log_error "Failed to convert '$wav_file'. Check ffmpeg output above if any."
    # Optionally remove partially created pcm file on error
    # rm -f "$output_pcm_file"
    ((error_count++))
  fi
  echo "---"
done

echo "----------------------------------------"
log_info "Conversion process finished."
log_info "Successfully converted: $converted_count files."
log_info "Errors during conversion: $error_count files."

if [ "$converted_count" -eq 0 ] && [ "$error_count" -eq 0 ]; then
    log_info "No .wav files found in '$INPUT_DIR' or all were skipped."
fi

exit 0