# /content/modules/img2img.py

import random
import sys
import numpy as np
from PIL import Image
import gradio as gr
import torch
from diffusers import DiffusionPipeline
from modules import pipeline
from modules import pipeline as pipe_module
from modules.pipeline import load_pipeline_img2img

# Define the function that ensures a number is divisible by 8 and close to the input value
def closest_divisible_by_8(number: int) -> int:
    remainder = number % 8
    if remainder == 0:
        return number
    lower_nearest = number - remainder
    higher_nearest = number + (8 - remainder)
    # Return the nearest number divisible by 8
    if abs(number - lower_nearest) <= abs(number - higher_nearest):
        return lower_nearest
    else:
        return higher_nearest

def img2img(prompt_i2i, negative_prompt_i2i, image_input_i2i, height_i2i, width_i2i, num_inference_steps_i2i, guidance_scale_i2i, strength_i2i, batch_count_i2i, seed_int="", scheduler=None):
    # Validate the seed input
    if seed_int == "":
        seed = random.randint(0, sys.maxsize)
    else:
        try:
            seed = int(seed_int)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return None

    torch.manual_seed(seed)

    global pipeline
    if pipeline is None:
        return "Pipeline is not loaded. Please click 'Load Pipeline' first."

    # Ensure height and width are divisible by 8
    height_i2i = closest_divisible_by_8(height_i2i)
    width_i2i = closest_divisible_by_8(width_i2i)
    
    # Convert the input NumPy array to a PIL image and convert to RGB format
    image_input_i2i = Image.fromarray(image_input_i2i).convert("RGB")

    # Generate images using the pipeline
    images = pipeline(prompt=prompt_i2i, negative_prompt=negative_prompt_i2i, image=image_input_i2i, height=height_i2i, width=width_i2i, num_inference_steps=num_inference_steps_i2i, guidance_scale=guidance_scale_i2i, strength=strength_i2i, num_images_per_prompt=batch_count_i2i).images

    # Convert generated images to PIL image format
    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]

    # Construct metadata string
    metadata_str = f" Seed: {seed}, Prompt: {prompt_i2i}, Negative Prompt: {negative_prompt_i2i}, Height: {height_i2i}, Width: {width_i2i}, Num Inference Steps: {num_inference_steps_i2i}, Guidance Scale: {guidance_scale_i2i}, Strength: {strength_i2i}"

    return images_pil, images_pil, metadata_str
