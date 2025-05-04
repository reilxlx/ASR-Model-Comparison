import os
import tempfile
import logging
from flask import Flask, request, jsonify, Response # Import Response
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from werkzeug.utils import secure_filename
import json

# --- Configuration ---
MODEL_DIR = "iic/SenseVoiceSmall"
DEVICE = "cuda:0" # Or "cpu"
VAD_MAX_SINGLE_SEGMENT_TIME = 30000 # milliseconds
BATCH_SIZE_S = 60
MERGE_LENGTH_S = 15
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'pcm', 'm4a'}

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json;charset=utf-8'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
                    handlers=[logging.StreamHandler()])

# --- Load FunASR Model ---
logging.info(f"Loading FunASR model: {MODEL_DIR} on device: {DEVICE}...")
try:
    model = AutoModel(
        model=MODEL_DIR,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": VAD_MAX_SINGLE_SEGMENT_TIME},
        device=DEVICE,
    )
    logging.info("FunASR model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load FunASR model: {e}", exc_info=True)
    model = None

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoint ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if model is None:
         # Use manual response for errors too, ensuring consistency
         error_payload = json.dumps({"error": "Model not loaded. Server initialization failed."}, ensure_ascii=False)
         return Response(error_payload, status=500, mimetype='application/json;charset=utf-8')

    if 'audio_file' not in request.files:
        error_payload = json.dumps({"error": "No audio_file part in the request"}, ensure_ascii=False)
        return Response(error_payload, status=400, mimetype='application/json;charset=utf-8')

    file = request.files['audio_file']

    if file.filename == '':
        error_payload = json.dumps({"error": "No selected file"}, ensure_ascii=False)
        return Response(error_payload, status=400, mimetype='application/json;charset=utf-8')


    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            file.save(temp_file.name)
            temp_filepath = temp_file.name
            temp_file.close()

            logging.info(f"Processing file: {filename} (saved to temp: {temp_filepath})")

            language_param = request.args.get('language', 'auto')
            logging.info(f"Using language: {language_param}")
            use_itn_param = request.args.get('use_itn', 'true').lower() == 'true'
            logging.info(f"Using ITN: {use_itn_param}")

            res = model.generate(
                input=temp_filepath,
                cache={},
                language=language_param,
                use_itn=use_itn_param,
                batch_size_s=BATCH_SIZE_S,
                merge_vad=True,
                merge_length_s=MERGE_LENGTH_S,
            )
            logging.info(f"Raw model result: {res}")

            if res and isinstance(res, list) and len(res) > 0 and res[0].get("text") is not None:
                 raw_text_from_model = res[0]["text"]
                 # CRITICAL LOG 1: See the exact raw string
                 logging.info(f"RAW TEXT FROM MODEL: {repr(raw_text_from_model)}")

                 processed_text = None # Initialize variable

                 # --- Attempt 1: Fix Incorrect Encoding (UTF-8 bytes as Latin-1) ---
                 try:
                     utf8_bytes = raw_text_from_model.encode('latin-1')
                     correctly_decoded_text = utf8_bytes.decode('utf-8')
                     # CRITICAL LOG 2: See if this decoding worked
                     logging.info(f"TEXT AFTER latin-1 -> utf-8 decode: {repr(correctly_decoded_text)}")
                     processed_text = correctly_decoded_text # Use this if successful
                 except Exception as decode_err:
                     logging.warning(f"Latin-1 -> UTF-8 decoding FAILED: {decode_err}. Falling back to raw text.")
                     # Fallback: Use the raw text directly if decoding fails
                     processed_text = raw_text_from_model


                 # CRITICAL LOG 3: Log text BEFORE postprocessing
                 logging.info(f"TEXT BEFORE rich_transcription_postprocess: {repr(processed_text)}")

                 # Apply rich text postprocessing ONLY if ITN was used
                 if use_itn_param:
                    try:
                        final_text = rich_transcription_postprocess(processed_text)
                        # CRITICAL LOG 4: Log text AFTER postprocessing
                        logging.info(f"TEXT AFTER rich_transcription_postprocess: {repr(final_text)}")
                    except Exception as postprocess_err:
                        logging.error(f"Error during rich_transcription_postprocess: {postprocess_err}", exc_info=True)
                        # Handle error during postprocessing, maybe return text before postprocessing?
                        final_text = processed_text # Fallback to text before postprocessing
                        logging.warning("Using text before postprocessing due to error.")
                 else:
                    final_text = processed_text # Use the text directly if ITN is off
                    logging.info("Skipping rich_transcription_postprocess as ITN is false.")


                 # CRITICAL LOG 5: Log the final string payload
                 logging.info(f"FINAL TEXT for JSON payload: {repr(final_text)}")

                 # --- Construct Response Manually ---
                 # Create the dictionary payload
                 payload_dict = {"transcription": final_text}
                 # Dump the dictionary to a JSON string, explicitly ensuring UTF-8
                 json_payload = json.dumps(payload_dict, ensure_ascii=False)
                 # Create the Flask Response object
                 response = Response(json_payload, mimetype='application/json;charset=utf-8')
                 # --- End Manual Response ---

                 logging.info(f"Sending Response with mimetype: {response.mimetype}")
                 return response

            # ... (rest of error handling using manual Response for consistency) ...
            elif res and isinstance(res, list) and len(res) > 0 and "timestamp" in res[0]:
                 logging.warning(f"Transcription produced data but no 'text' field for {filename}. Result: {res}")
                 payload_dict = {"transcription": "", "info": "No text transcribed, possibly empty audio segment.", "raw_result": res}
                 json_payload = json.dumps(payload_dict, ensure_ascii=False)
                 return Response(json_payload, status=200, mimetype='application/json;charset=utf-8') # 200 OK, just no text
            else:
                 logging.warning(f"Transcription resulted in empty or invalid data for {filename}. Result: {res}")
                 payload_dict = {"error": "Transcription failed or produced no text", "raw_result": res}
                 json_payload = json.dumps(payload_dict, ensure_ascii=False)
                 return Response(json_payload, status=500, mimetype='application/json;charset=utf-8')

        except Exception as e:
            logging.error(f"Critical error during transcription processing for {filename}: {e}", exc_info=True)
            error_message = f"An error occurred during processing: {str(e)}"
            payload_dict = {"error": error_message}
            json_payload = json.dumps(payload_dict, ensure_ascii=False)
            return Response(json_payload, status=500, mimetype='application/json;charset=utf-8')
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                    logging.info(f"Cleaned up temporary file: {temp_file.name}")
                except OSError as e:
                    logging.error(f"Error removing temporary file {temp_file.name}: {e}")
    else:
        error_payload = json.dumps({"error": f"File type not allowed. Allowed types: {list(ALLOWED_EXTENSIONS)}"}, ensure_ascii=False)
        return Response(error_payload, status=400, mimetype='application/json;charset=utf-8')


# --- Run the Server ---
if __name__ == '__main__':
    # Consider using a proper WSGI server like gunicorn for production
    # gunicorn --bind 0.0.0.0:5000 app:app --workers 4 --threads 2
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)