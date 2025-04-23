from flask import Blueprint, render_template, request
from diffusers import StableDiffusionPipeline
import os
import uuid
import torch

# Initialize Blueprint
image_blueprint = Blueprint("image", __name__)

# Directory for storing generated images
IMAGE_DIR = "static/generated_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Initialize Stable Diffusion Pipeline
image_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
image_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def clear_cuda_cache():
    """Clear CUDA cache to free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

@image_blueprint.route("/gen_image", methods=["GET", "POST"])
def generate_image():
    """Handle image generation requests."""
    if request.method == "POST":
        prompt = request.form["prompt"]
        image_paths = []
        for _ in range(2):  # Generate 2 images
            try:
                print(f"Generating image for prompt: {prompt}")
                image = image_pipe(prompt).images[0]
                filename = f"{uuid.uuid4()}.png"
                image_path = os.path.join(IMAGE_DIR, filename)
                image.save(image_path)
                image_paths.append(filename)
            except Exception as e:
                print(f"Error during image generation: {e}")
                return render_template("image.html", images=[], error="Error generating images.")
        clear_cuda_cache()  # Clear GPU memory after image generation
        return render_template("image.html", images=image_paths)
    return render_template("image.html", images=[])