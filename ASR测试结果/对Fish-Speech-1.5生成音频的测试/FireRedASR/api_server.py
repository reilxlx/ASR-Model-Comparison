import os
import uuid
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from fireredasr.models.fireredasr import FireRedAsr

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'} # Adjust based on supported formats
MODEL_VARIANT = "aed"
# IMPORTANT: Adjust this path to your actual model location
MODEL_PATH = "pretrained_models/FireRedASR-AED-L" 
DEFAULT_GPU_USE = 1 # Use 1 for GPU, 0 for CPU
DEFAULT_BEAM_SIZE = 3
DEFAULT_NBEST = 1
DEFAULT_DECODE_MAX_LEN = 0
DEFAULT_SOFTMAX_SMOOTHING = 1.25
DEFAULT_AED_LENGTH_PENALTY = 0.6
DEFAULT_EOS_PENALTY = 1.0

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Limit uploads to 32MB, adjust as needed

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model (Load once on startup) ---
print("Loading FireRedASR model...")
try:
    model = FireRedAsr.from_pretrained(MODEL_VARIANT, MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if model loading fails, as the API is useless without it
    exit(1) 

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoint ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    API endpoint to transcribe audio.
    Accepts POST requests with either:
    1. JSON body: {"uttid": "some_id", "wav_path": "/path/to/audio/on/server.wav"}
    2. multipart/form-data: With 'audio_file' (the audio file) and optional 'uttid' field.
    """
    start_time = time.time()
    
    uttid = None
    wav_path = None
    uploaded_file_path = None # Keep track if we need to delete a temporary file

    try:
        # --- Input Handling ---
        if request.is_json:
            # Method 1: JSON body with server-side path
            data = request.get_json()
            if not data or 'wav_path' not in data:
                return jsonify({"error": "Missing 'wav_path' in JSON body"}), 400
            
            uttid = data.get('uttid', f"utt_{uuid.uuid4()}") # Generate uttid if not provided
            wav_path = data['wav_path']

            # Security check: Basic check to prevent accessing arbitrary files
            # For production, you might want more robust path validation
            if ".." in wav_path or not os.path.isabs(wav_path):
                 # Allowing relative paths might be okay if contained within a specific data root
                 # For simplicity here, we require absolute path or ensure it's within allowed area.
                 # Let's check if it exists relative to CWD, otherwise reject relative.
                 if not os.path.exists(wav_path):
                     # A more robust check would involve comparing against a defined MEDIA_ROOT
                      return jsonify({"error": f"Invalid or non-absolute path specified: {wav_path}. Or file does not exist."}), 400
            
            if not os.path.exists(wav_path):
                return jsonify({"error": f"File not found on server: {wav_path}"}), 404

        elif request.files:
             # Method 2: File Upload
            if 'audio_file' not in request.files:
                return jsonify({"error": "Missing 'audio_file' in form data"}), 400

            file = request.files['audio_file']
            uttid = request.form.get('uttid', f"utt_{uuid.uuid4()}") # Get uttid from form or generate

            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(f"{uttid}_{file.filename}") # Use uttid in filename
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(uploaded_file_path)
                wav_path = uploaded_file_path # Use the saved path for transcription
                print(f"Uploaded file saved to: {wav_path}")
            else:
                return jsonify({"error": f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"}), 400
        else:
            return jsonify({"error": "Invalid request. Use JSON or multipart/form-data with 'audio_file'"}), 400

        # --- Transcription ---
        if not uttid or not wav_path:
             # This case should ideally be caught earlier, but as a safeguard:
            return jsonify({"error": "Internal error: uttid or wav_path not set"}), 500

        batch_uttid = [uttid]
        batch_wav_path = [wav_path]

        # Get parameters from request or use defaults (can be extended)
        # For simplicity, we use defaults here. You could parse request.args or request.json
        # to allow overriding these per request.
        transcribe_params = {
            "use_gpu": int(request.args.get('use_gpu', DEFAULT_GPU_USE)),
            "beam_size": int(request.args.get('beam_size', DEFAULT_BEAM_SIZE)),
            "nbest": int(request.args.get('nbest', DEFAULT_NBEST)),
            "decode_max_len": int(request.args.get('decode_max_len', DEFAULT_DECODE_MAX_LEN)),
            "softmax_smoothing": float(request.args.get('softmax_smoothing', DEFAULT_SOFTMAX_SMOOTHING)),
            "aed_length_penalty": float(request.args.get('aed_length_penalty', DEFAULT_AED_LENGTH_PENALTY)),
            "eos_penalty": float(request.args.get('eos_penalty', DEFAULT_EOS_PENALTY))
        }
        
        print(f"Transcribing {wav_path} with params: {transcribe_params}")

        # Perform transcription
        results = model.transcribe(
            batch_uttid,
            batch_wav_path,
            transcribe_params
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Transcription successful in {processing_time:.4f} seconds.")

        # --- Response ---
        if results:
            # Add processing time to the result if desired
            results[0]['processing_time_seconds'] = f"{processing_time:.4f}"
            return jsonify(results[0]) # Return the first (and only) result directly
        else:
            return jsonify({"error": "Transcription failed or returned no results"}), 500

    except Exception as e:
        print(f"Error during transcription request: {e}")
        # Log the full traceback for debugging if needed
        import traceback
        traceback.print_exc() 
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

    finally:
        # --- Cleanup Uploaded File ---
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"Cleaned up temporary file: {uploaded_file_path}")
            except Exception as e:
                print(f"Error deleting temporary file {uploaded_file_path}: {e}")


# --- Run Flask App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible from other machines on the network
    # debug=True is helpful during development but should be False in production
    app.run(host='0.0.0.0', port=5000, debug=True) 