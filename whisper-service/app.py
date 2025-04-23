from flask import Flask, request, jsonify
import base64
from flask_cors import CORS
import torch
from faster_whisper import WhisperModel
import os
import logging
import time
import uuid
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("whisper_transcription.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Initialize Whisper model at startup
logger.info("Loading Whisper transcription model...")
model = WhisperModel(
    model_size_or_path="base",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)
logger.info("Whisper model loaded successfully")

def process_audio_with_whisper(wav_file_path, request_id=None):
    """Process audio file with Whisper for transcription with word-level timestamps"""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    
    start_time = time.time()
    logger.info(f"[WHISPER-{request_id}] Starting audio transcription")
    
    # Log file details
    try:
        file_size = os.path.getsize(wav_file_path)
        logger.info(f"[WHISPER-{request_id}] Processing audio file: {wav_file_path}")
        logger.info(f"[WHISPER-{request_id}] File size: {file_size/1024:.2f} KB")
    except Exception as e:
        logger.error(f"[WHISPER-{request_id}] Error getting file info: {str(e)}")
    
    try:
        # Track GPU memory if available
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            logger.info(f"[WHISPER-{request_id}] Initial GPU memory: {memory_before/1024**2:.2f} MB")
        
        # Start transcription
        logger.info(f"[WHISPER-{request_id}] Starting transcription with beam_size=5...")
        transcription_start = time.time()
        
        segments, info = model.transcribe(
            wav_file_path,
            beam_size=5,
            word_timestamps=True
        )
        
        # Convert generator to list
        segments = list(segments)
        transcription_time = time.time() - transcription_start
        
        # Log transcription results
        logger.info(f"[WHISPER-{request_id}] ✅ Transcription completed in {transcription_time:.2f}s")
        logger.info(f"[WHISPER-{request_id}] Detected language: {info.language} (probability: {info.language_probability:.4f})")
        logger.info(f"[WHISPER-{request_id}] Number of segments: {len(segments)}")
        
        # Process word-level results
        logger.info(f"[WHISPER-{request_id}] Processing word-level timestamps...")
        word_level_results = []
        total_words = 0
        
        for segment_idx, segment in enumerate(segments):
            segment_words = list(segment.words)
            total_words += len(segment_words)
            
            logger.info(f"[WHISPER-{request_id}] Segment {segment_idx+1}: {len(segment_words)} words, {segment.start:.2f}s to {segment.end:.2f}s")
            logger.info(f"[WHISPER-{request_id}] Text: \"{segment.text}\"")
            
            for word in segment_words:
                word_level_results.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2)
                })
        
        logger.info(f"[WHISPER-{request_id}] Total words detected: {total_words}")
        
        # Process line-level results
        logger.info(f"[WHISPER-{request_id}] Processing line-level grouping...")
        line_level_results = []
        current_line = []
        current_text = []
        
        # Parameters for line grouping
        MAX_CHARS = 30
        MAX_DURATION = 2.5
        MAX_GAP = 1.5
        
        logger.info(f"[WHISPER-{request_id}] Line grouping parameters: MAX_CHARS={MAX_CHARS}, MAX_DURATION={MAX_DURATION}s, MAX_GAP={MAX_GAP}s")
        
        for idx, word in enumerate(word_level_results):
            current_line.append(word)
            current_text.append(word["word"])
            
            line_text = " ".join(current_text)
            line_duration = word["end"] - current_line[0]["start"]
            
            should_break = False
            
            if len(line_text) > MAX_CHARS:
                should_break = True
            elif line_duration > MAX_DURATION:
                should_break = True
            elif idx > 0:
                gap = word["start"] - word_level_results[idx-1]["end"]
                if gap > MAX_GAP:
                    should_break = True
            
            if should_break or idx == len(word_level_results) - 1:
                if current_line:
                    new_line = {
                        "text": " ".join(current_text),
                        "start": current_line[0]["start"],
                        "end": current_line[-1]["end"],
                        "words": current_line.copy()
                    }
                    
                    line_level_results.append(new_line)
                    current_line = []
                    current_text = []
        
        logger.info(f"[WHISPER-{request_id}] Created {len(line_level_results)} line groups")
        
        # Log sample lines for verification
        if line_level_results:
            sample_lines = min(3, len(line_level_results))
            for i in range(sample_lines):
                logger.info(f"[WHISPER-{request_id}] Sample line {i+1}: \"{line_level_results[i]['text']}\"")
        
        # Prepare results
        results = {
            "word_level": word_level_results,
            "line_level": line_level_results,
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "processing_time": {
                "transcription": transcription_time,
                "total": time.time() - start_time
            }
        }
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            logger.info(f"[WHISPER-{request_id}] GPU memory after processing: {memory_after/1024**2:.2f} MB")
            logger.info(f"[WHISPER-{request_id}] Memory difference: {(memory_after-memory_before)/1024**2:.2f} MB")
        
        logger.info(f"[WHISPER-{request_id}] ✓ Transcription processing completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"[WHISPER-{request_id}] ❌ ERROR in transcription: {str(e)}")
        logger.error(f"[WHISPER-{request_id}] Exception type: {type(e).__name__}")
        raise
    
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[WHISPER-{request_id}] GPU memory cleaned up")

@app.route('/process_audio', methods=["POST"])
def process_audio():
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[WHISPER-{request_id}] Received audio processing request")
    
    # Extract and validate audio data
    data = request.json
    if 'audio_data' not in data:
        logger.error(f"[WHISPER-{request_id}] ERROR: No audio data in request")
        return jsonify({"error": "No audio data provided"}), 400
    
    audio_base64 = data['audio_data']
    audio_data_length = len(audio_base64)
    logger.info(f"[WHISPER-{request_id}] Audio data size: {audio_data_length/1024:.2f} KB")
    
    # Create temporary file
    temp_path = f"temp_audio_{os.getpid()}_{request_id}.wav"
    logger.info(f"[WHISPER-{request_id}] Saving to temporary file: {temp_path}")
    
    try:
        # Decode and save audio
        audio_bytes = base64.b64decode(audio_base64)
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"[WHISPER-{request_id}] File created: {file_size/1024:.2f} KB")
        
        # Process the audio
        results = process_audio_with_whisper(temp_path, request_id)
        
        # Log summary
        word_count = len(results["word_level"])
        line_count = len(results["line_level"])
        
        logger.info(f"[WHISPER-{request_id}] Results summary:")
        logger.info(f"[WHISPER-{request_id}] - Words detected: {word_count}")
        logger.info(f"[WHISPER-{request_id}] - Lines created: {line_count}")
        logger.info(f"[WHISPER-{request_id}] - Language: {results['detected_language']} (probability: {results['language_probability']:.4f})")
        
        if word_count > 0:
            first_word = results["word_level"][0]
            last_word = results["word_level"][-1]
            audio_duration = last_word["end"]
            logger.info(f"[WHISPER-{request_id}] - Audio duration: {audio_duration:.2f}s")
            logger.info(f"[WHISPER-{request_id}] - Words per second: {word_count/audio_duration:.2f}")
        
        total_time = time.time() - start_time
        logger.info(f"[WHISPER-{request_id}] ✓ Total processing time: {total_time:.2f}s")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"[WHISPER-{request_id}] ❌ ERROR during request handling: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[WHISPER-{request_id}] Temporary file removed")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        "status": "healthy",
        "service": "whisper-transcription",
        "model": "base",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        status["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"
        status["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved()/1024**2:.2f} MB"
    
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting whisper transcription service on port {port}")
    app.run(host='0.0.0.0', port=port)