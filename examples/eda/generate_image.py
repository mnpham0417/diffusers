import os
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

import diffusers
# print(diffusers.__file__)
# assert 0

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"

# Define the directory to save images
save_dir = "/scratch/mp5847/eda/starry_night_sd1.4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a generator and set the seed
generator = torch.Generator(device=device)
generator.manual_seed(42)

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to(device)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "latents"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "noise_preds"), exist_ok=True)

num_generations = 50
for i in range(num_generations):
    # captions = ["a painting in the style of Monet"]
    # captions = ["a photo of a church"]
    captions = ["Starry Night"]

    # Generate an image
    if device == "cuda":
        with torch.cuda.amp.autocast():
            out = pipeline(captions, guidance_scale=7.5, generator=generator, safety_checker=None)
    else:
        out = pipeline(captions, guidance_scale=7.5, generator=generator, safety_checker=None)

    generated_images = out.images
    latents = out.latents
    noise_preds = out.noise_preds
    
    latents = np.array(latents)
    noise_preds = np.array(noise_preds)

    # Save the generated image
    for generated_image in generated_images:
        image_filename = os.path.join(save_dir, "images", f"{captions[0].replace(' ', '_')}_{i}.png")
        generated_image.save(image_filename)
        print(f"Saved image to {image_filename}")
        
    latent_filename = os.path.join(save_dir, "latents", f"{captions[0].replace(' ', '_')}_{i}.npy")
    np.save(latent_filename, latents)
    print(f"Saved latent to {latent_filename}")

    noise_pred_filename = os.path.join(save_dir, "noise_preds", f"{captions[0].replace(' ', '_')}_{i}.npy")
    np.save(noise_pred_filename, noise_preds)
    print(f"Saved noise predictions to {noise_pred_filename}")
        
