from flask import Flask, request, jsonify
import base64
from io import BytesIO
from flask_cors import CORS
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
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
        logging.FileHandler("image_generation.log"),
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

# Initialize the model at startup
logger.info("Loading image generation model (DreamShaper XL)...")
pipe = AutoPipelineForText2Image.from_pretrained(
    'lykon/dreamshaper-xl-lightning', 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.safety_checker = None  # Disable safety checker for faster inference
logger.info("Image generation model loaded successfully")

def generate_image_from_prompt(prompt, inference_steps=8, guidance_scale=7.5, request_id=None):
    """Generate image using Stable Diffusion based on text prompt"""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    
    start_time = time.time()
    logger.info(f"[IMAGE-{request_id}] Generating image for prompt: {prompt[:100]}..." if len(prompt) > 100 else f"[IMAGE-{request_id}] Generating image for prompt: {prompt}")
    logger.info(f"[IMAGE-{request_id}] Parameters: steps={inference_steps}, guidance_scale={guidance_scale}")
    
    try:
        # Track GPU memory if available
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            logger.info(f"[IMAGE-{request_id}] Initial GPU memory: {memory_before/1024**2:.2f} MB")
        
        # Set reproducible seed
        generator = torch.manual_seed(0)
        
        # Generate the image
        generation_start = time.time()
        with torch.autocast(device):
            generated_image = pipe(
                prompt,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        generation_time = time.time() - generation_start
        logger.info(f"[IMAGE-{request_id}] ✅ Generation completed in {generation_time:.2f}s")
        logger.info(f"[IMAGE-{request_id}] Image dimensions: {generated_image.size}")
        
        # Convert to base64
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Log response size
        logger.info(f"[IMAGE-{request_id}] Response size: {len(img_str)/1024:.2f} KB")
        
        # Track GPU memory after processing
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            logger.info(f"[IMAGE-{request_id}] GPU memory after generation: {memory_after/1024**2:.2f} MB")
            logger.info(f"[IMAGE-{request_id}] Memory difference: {(memory_after-memory_before)/1024**2:.2f} MB")
        
        total_time = time.time() - start_time
        logger.info(f"[IMAGE-{request_id}] ✓ Total processing time: {total_time:.2f}s")
        
        return img_str
        
    except Exception as e:
        logger.error(f"[IMAGE-{request_id}] ❌ ERROR: {str(e)}")
        logger.error(f"[IMAGE-{request_id}] Exception type: {type(e).__name__}")
        raise
    
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[IMAGE-{request_id}] GPU cache cleared")

@app.route('/generate-image', methods=['POST'])
def generate_image():
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[IMAGE-{request_id}] Received image generation request")
    
    # Extract and validate the prompt
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        logger.error(f"[IMAGE-{request_id}] ERROR: No prompt provided")
        return jsonify({"error": "Prompt is required"}), 400
    
    # Extract additional parameters if provided, or use defaults
    inference_steps = data.get('inference_steps', 8)
    guidance_scale = data.get('guidance_scale', 7.5)
    
    try:
        # Generate the image
        img_str = generate_image_from_prompt(
            prompt, 
            inference_steps, 
            guidance_scale, 
            request_id
        )
        
        # Return the base64-encoded image
        return jsonify({"image_data": img_str})
        
    except Exception as e:
        logger.error(f"[IMAGE-{request_id}] Request failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        "status": "healthy",
        "service": "image-generation",
        "model": "dreamshaper-xl-lightning",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        status["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"
        status["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved()/1024**2:.2f} MB"
    
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting image generation service on port {port}")
    app.run(host='0.0.0.0', port=port)