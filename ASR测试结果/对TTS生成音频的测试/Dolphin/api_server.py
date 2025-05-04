import dolphin
import flask
from flask import Flask, request, Response # Import Response
import tempfile
import os
import re
import logging
import json # Import json

# --- Configuration ---
MODEL_NAME = "base"
# Adjust this path if your cache/model location is different
MODEL_PATH = "/root/.cache/modelscope/hub/models/DataoceanAI/dolphin-base"
DEVICE = "cuda" # Or "cpu" if you don't have a GPU or want to use CPU
HOST = '0.0.0.0' # Listen on all network interfaces
PORT = 5001      # Port for the API server

# --- Initialize Flask App ---
app = Flask(__name__)
# While JSON_AS_ASCII=False should work, we'll bypass jsonify for more control
# app.config['JSON_AS_ASCII'] = False # Keep or remove, shouldn't matter with manual response
logging.basicConfig(level=logging.INFO)

# --- Load Model (Load only once on startup) ---
model = None
try:
    app.logger.info(f"Loading Dolphin model '{MODEL_NAME}' from {MODEL_PATH} onto {DEVICE}...")
    model = dolphin.load_model(MODEL_NAME, MODEL_PATH, DEVICE)
    app.logger.info("Dolphin model loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load Dolphin model: {e}")
    # import sys
    # sys.exit("Model loading failed. Exiting.")

# --- Regex for extracting text ---
ASR_TEXT_PATTERN = re.compile(r"<asr><\d+\.\d+>\s*(.*?)\s*<\d+\.\d+>", re.IGNORECASE)

# --- Helper function to create JSON Response ---
def create_json_response(data, status_code=200):
    """Creates a Flask Response object with JSON data and UTF-8 charset."""
    json_string = json.dumps(data, ensure_ascii=False)
    response = Response(
        json_string,
        status=status_code,
        mimetype='application/json; charset=utf-8' # Explicitly set charset
    )
    return response

# --- API Endpoint ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    API endpoint to transcribe an audio file using the Dolphin model.
    Expects a POST request with multipart/form-data containing an audio file
    under the key 'audio_file'.
    """
    if model is None:
         app.logger.error("Model not loaded, cannot process request.")
         return create_json_response({"error": "Model not loaded or failed to load"}, 500)

    if 'audio_file' not in request.files:
        app.logger.warning("No 'audio_file' part in the request.")
        return create_json_response({"error": "No audio file provided in the 'audio_file' field"}, 400)

    file = request.files['audio_file']

    if file.filename == '':
        app.logger.warning("No selected file.")
        return create_json_response({"error": "No selected file"}, 400)

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        file.save(temp_path)
        app.logger.info(f"Temporary audio file saved to: {temp_path}")

        app.logger.info(f"Loading audio waveform from: {temp_path}")
        waveform = dolphin.load_audio(temp_path)

        app.logger.info("Starting transcription...")
        result = model(waveform, lang_sym="zh", region_sym="CN")
        app.logger.info(f"Raw ASR result: {result.text}")

        match = ASR_TEXT_PATTERN.search(result.text)
        if match:
            extracted_text = match.group(1).strip()
            app.logger.info(f"Extracted text: {extracted_text}")
            # Use the helper function to create the response
            return create_json_response({"transcription": extracted_text})
        else:
            app.logger.warning(f"Could not extract text from ASR result: {result.text}")
            return create_json_response({
                "error": "Could not parse transcription from ASR output",
                "raw_output": result.text
             }, 500)

    except Exception as e:
        app.logger.error(f"An error occurred during transcription: {e}", exc_info=True)
        return create_json_response({"error": f"An internal error occurred: {str(e)}"}, 500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            app.logger.info(f"Removed temporary audio file: {temp_path}")
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            app.logger.info(f"Removed temporary directory: {temp_dir}")

# --- Run the App ---
if __name__ == '__main__':
    app.logger.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)