from flask import Flask, request, jsonify
import base64
from io import BytesIO
from flask_cors import CORS
import torch
from transformers import BarkModel, AutoProcessor
import scipy.io.wavfile
import logging
import time
import uuid
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("voice_generation.log"),
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

# Initialize the Bark model and processor at startup
logger.info("Loading Bark text-to-speech model...")
bark_model = BarkModel.from_pretrained(
    "suno/bark",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

# Try to optimize with BetterTransformer if available
try:
    bark_model = bark_model.to_bettertransformer()
    logger.info("BetterTransformer optimization enabled")
except Exception as e:
    logger.info(f"BetterTransformer optimization not available: {e}")

# Load processor
processor = AutoProcessor.from_pretrained("suno/bark")
logger.info("Bark model and processor loaded successfully")

def generate_speech_with_bark(text, voice_preset, request_id=None):
    """Generate speech using Bark with consistent voice"""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    
    start_time = time.time()
    
    # Preview the text for logging
    text_preview = text[:100] + "..." if len(text) > 100 else text
    logger.info(f"[BARK-{request_id}] Generating speech for: \"{text_preview}\"")
    logger.info(f"[BARK-{request_id}] Text length: {len(text)} characters")
    logger.info(f"[BARK-{request_id}] 🎙️ Using voice preset: {voice_preset}")
    
    try:
        # Track GPU memory if available
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            logger.info(f"[BARK-{request_id}] Initial GPU memory: {memory_before/1024**2:.2f} MB")
        
        # Generate speech with the specified voice preset
        logger.info(f"[BARK-{request_id}] Processing text input...")
        inference_start = time.time()
        
        inputs = processor(text, voice_preset=voice_preset)
        logger.info(f"[BARK-{request_id}] Input prepared, moving to device")
        
        # Move inputs to the correct device if they're tensors
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        logger.info(f"[BARK-{request_id}] Starting inference...")
        
        with torch.inference_mode():
            speech_output = bark_model.generate(**inputs)
            
        inference_time = time.time() - inference_start
        logger.info(f"[BARK-{request_id}] ✅ Speech generation completed in {inference_time:.2f}s")
        
        # Process the audio output
        audio_data = speech_output[0].cpu().numpy()
        audio_data = (audio_data * 32767).astype('int16')
        sample_rate = bark_model.generation_config.sample_rate
        
        # Log audio properties
        audio_duration = len(audio_data) / sample_rate
        logger.info(f"[BARK-{request_id}] Generated audio duration: {audio_duration:.2f}s")
        logger.info(f"[BARK-{request_id}] Audio sample rate: {sample_rate}Hz")
        
        # Convert to base64
        buffer = BytesIO()
        scipy.io.wavfile.write(
            buffer,
            rate=sample_rate,
            data=audio_data
        )
        audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Log memory statistics
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            logger.info(f"[BARK-{request_id}] GPU memory after inference: {memory_after/1024**2:.2f} MB")
            logger.info(f"[BARK-{request_id}] Memory difference: {(memory_after-memory_before)/1024**2:.2f} MB")
        
        total_time = time.time() - start_time
        logger.info(f"[BARK-{request_id}] ✓ Total processing time: {total_time:.2f}s")
        
        # Return the audio data and metadata
        return {
            "audio_data": audio_b64,
            "content_type": "audio/wav",
            "duration": audio_duration,
            "sample_rate": sample_rate
        }
        
    except Exception as e:
        logger.error(f"[BARK-{request_id}] ❌ ERROR in generation: {str(e)}")
        logger.error(f"[BARK-{request_id}] Exception type: {type(e).__name__}")
        raise
    
    finally:
        # Clean up GPU resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[BARK-{request_id}] GPU memory cleaned up")

@app.route('/generate_sound', methods=['POST'])
def generate_sound():
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[BARK-{request_id}] Received sound generation request")
    
    # Extract and validate parameters
    data = request.json
    
    text = data.get('text')
    voice_preset = data.get('voice_type', 'v2/en_speaker_6')  # Default voice if not specified
    
    if not text:
        logger.error(f"[BARK-{request_id}] ERROR: No text provided")
        return jsonify({"error": "Text is required"}), 400
    
    try:
        # Generate speech audio
        result = generate_speech_with_bark(text, voice_preset, request_id)
        
        # Return the generation result
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[BARK-{request_id}] Request failed: {str(e)}")
        return jsonify({"error": f"Speech generation failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        "status": "healthy",
        "service": "voice-generation",
        "model": "suno/bark",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        status["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"
        status["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved()/1024**2:.2f} MB"
    
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting voice generation service on port {port}")
    app.run(host='0.0.0.0', port=port)