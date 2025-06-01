#!/bin/bash

# --- Configuration ---
# Base FFmpeg options for 16kHz, 16-bit, mono PCM (s16le)
FFMPEG_OPTS_PCM_16K_S16LE_MONO="-ar 16000 -ac 1 -acodec pcm_s16le -f s16le"
# FFmpeg audio filter to add 0.5 seconds of silence at the end
AUDIO_FILTER_PAD_SILENCE="-af \"apad=pad_dur=0.5\""

# --- Helper Functions ---
usage() {
  echo "Usage: $0 <input_wav_directory> <output_pcm_directory>"
  echo "Converts .wav files to .pcm (16kHz, mono, 16-bit) and adds 0.5s silence at the end."
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
log_info "Starting WAV to PCM conversion with silence padding..."
log_info "Input directory: $INPUT_DIR"
log_info "Output directory: $OUTPUT_DIR"
log_info "FFmpeg base options: $FFMPEG_OPTS_PCM_16K_S16LE_MONO"
log_info "FFmpeg audio filter for padding: $AUDIO_FILTER_PAD_SILENCE"
echo "----------------------------------------"

converted_count=0
# skipped_count=0 # Not actively used for skipping individual files in this version, but for overall "no files found"
error_count=0

# Find .wav files (case-insensitive) in the input directory (non-recursively)
# -print0 and read -d $'\0' handle filenames with spaces or special characters
find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.wav" -print0 | while IFS= read -r -d $'\0' wav_file; do
  filename_with_ext=$(basename "$wav_file")
  # Removes the last extension (e.g., .wav or .WAV)
  filename_no_ext="${filename_with_ext%.*}"
  output_pcm_file="$OUTPUT_DIR/${filename_no_ext}.pcm"

  log_info "Processing: '$wav_file'"

  # ffmpeg command
  # -y: Overwrite output files without asking
  # -v error: Show only errors. Use -v quiet -stats for progress. Use -v verbose for more details.
  # The AUDIO_FILTER_PAD_SILENCE is placed before other output options.
  # Note: Bash requires careful handling of quotes when variables contain spaces or special characters.
  # Here, AUDIO_FILTER_PAD_SILENCE contains quotes for the -af option's argument.
  # We will use eval to correctly parse the command string with nested quotes,
  # or construct the command as an array. Using an array is safer.

  ffmpeg_cmd=(ffmpeg -i "$wav_file")
  # Add filter options by splitting the string AUDIO_FILTER_PAD_SILENCE
  # This is a bit safer than eval for this specific case.
  # However, for complex filters, an array definition from the start is best.
  # For this specific filter, direct inclusion might also work if bash handles it.
  # Let's try direct inclusion first as it's simpler and often works.
  # If issues arise with complex filenames or filter arguments, an array approach is more robust.

  # Simpler approach for this specific filter string:
  # if ffmpeg -i "$wav_file" $AUDIO_FILTER_PAD_SILENCE $FFMPEG_OPTS_PCM_16K_S16LE_MONO "$output_pcm_file" -y -v error; then

  # More robust approach using an array for ffmpeg arguments:
  cmd_array=(ffmpeg -y -v error -i "$wav_file")
  # Add filter options: -af "apad=pad_dur=0.5"
  # The quotes around apad=pad_dur=0.5 are important for ffmpeg
  cmd_array+=(-af "apad=pad_dur=0.5")
  # Add PCM conversion options (split string into array elements)
  # shellcheck disable=SC2086
  cmd_array+=($FFMPEG_OPTS_PCM_16K_S16LE_MONO)
  cmd_array+=("$output_pcm_file")

  # log_info "Executing: ${cmd_array[*]}" # For debugging the command

  if "${cmd_array[@]}"; then
    log_info "Successfully converted and padded: '$output_pcm_file'"
    ((converted_count++))
  else
    log_error "Failed to convert/pad '$wav_file'. Check ffmpeg output above if any."
    # Optionally remove partially created pcm file on error
    # rm -f "$output_pcm_file"
    ((error_count++))
  fi
  echo "---"
done

echo "----------------------------------------"
log_info "Conversion process finished."
log_info "Successfully converted/padded: $converted_count files."
log_info "Errors during conversion/padding: $error_count files."

if [ "$converted_count" -eq 0 ] && [ "$error_count" -eq 0 ]; then
    log_info "No .wav files found in '$INPUT_DIR'."
fi

exit 0
