import os
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
import torch
from torch.utils.data import Dataset

# Load the Stable Diffusion pipeline
model_id = "/scratch/mp5847/robust-concept-erasure-checkpoints/vg_sd_v1.4_ascent"

# Define the directory to save images
save_dir = "./test/"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to(device)

os.makedirs(save_dir, exist_ok=True)

for i in range(5):
    captions = ["a painting in the style of Monet"]
    # Generate an image
    with torch.autocast(device):
        generated_images = pipeline(captions, guidance_scale=7.5, safety_checker=None).images

    # Save the generated image
    for _, generated_image in enumerate(generated_images):
        image_filename = os.path.join(save_dir, f"{captions[0]}_{i}.png")
        generated_image.save(image_filename)
        print(f"Saved image to {image_filename}")