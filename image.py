import os
import torch
import uuid
from flask import Flask, render_template, request, send_from_directory
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline

# Flask setup
app = Flask(__name__)

# Define model details
MODEL_REPO = "runwayml/stable-diffusion-v1-5"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")  # Default Hugging Face cache directory

# Directory to store generated images
IMAGE_DIR = "static/generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Check device availability
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16  # Use lower precision for faster GPU inference
else:
    device = "cpu"
    dtype = torch.float32  # Prevents errors on CPU

# Ensure model is downloaded
try:
    print("Checking and downloading model if necessary...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename="model_index.json",  # Ensures repo exists
        cache_dir=CACHE_DIR
    )
    print(f"Model repository verified: {MODEL_REPO}")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)

# Load Stable Diffusion v1.5 with error handling
try:
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_REPO, torch_dtype=dtype)
    pipe.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def generate_images(prompt, num_images=8):
    """Generate multiple images based on the given prompt."""
    image_paths = []
    for _ in range(num_images):
        try:
            print(f"Generating image for prompt: {prompt}")
            image = pipe(prompt).images[0]

            # Generate unique filename
            filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(IMAGE_DIR, filename)
            image.save(image_path)

            image_paths.append(filename)
        except Exception as e:
            print(f"Error during image generation: {e}")
    # Clear GPU memory after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    return image_paths

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]

        # Generate 8 images
        image_paths = generate_images(prompt, num_images=8)
        if not image_paths:
            return render_template("index.html", images=[], error="Error generating images.")
        
        # Add a cache-busting token (e.g., a UUID)
        cache_buster = str(uuid.uuid4())
        return render_template("index.html", images=image_paths, cache_buster=cache_buster)

    return render_template("index.html", images=[], cache_buster=None)

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(IMAGE_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)