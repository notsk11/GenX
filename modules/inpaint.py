# /content/modules/inpaint.py

import random
import sys
import numpy as np
from PIL import Image
import gradio as gr
import torch
from diffusers import DiffusionPipeline
from modules import pipeline
from modules import pipeline as pipe_module
from modules.pipeline import load_pipeline_inpaint

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

def inpaint(prompt_i2i, negative_prompt_i2i, image_input_inpaint, height_inpaint, width_inpaint, num_inference_steps_inpaint, guidance_scale_inpaint, strength_inpaint, batch_count_inpaint, seed_int="", scheduler=None):
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
    height_inpaint = closest_divisible_by_8(height_inpaint)
    width_inpaint = closest_divisible_by_8(width_inpaint)

    # Convert `batch_count_inpaint` to integer and handle empty string
    if batch_count_inpaint == "":
        batch_count_inpaint = 1  # Set a default value if empty
    else:
        try:
            batch_count_inpaint = int(batch_count_inpaint)
        except ValueError:
            print("Invalid input for batch count. Please enter a valid number.")
            return None

    # Check if input contains 'image' and 'mask' keys or if it's a single image
    if isinstance(image_input_inpaint, dict):
        image = image_input_inpaint.get('image').convert("RGB")
        mask_img = image_input_inpaint.get('mask').convert("RGB")
    else:
        # If input is a single image, you may need to adjust this part according to the expected use case
        image = image_input_inpaint.convert("RGB")
        mask_img = None

    # Convert other input parameters if necessary
    # Ensure `num_inference_steps_inpaint` and `guidance_scale_inpaint` are correctly typed

    # Generate images using the pipeline
    images = pipe_module.pipeline(prompt=prompt_i2i, negative_prompt=negative_prompt_i2i, image=image, mask_image=mask_img, height=height_inpaint, width=width_inpaint, num_inference_steps=num_inference_steps_inpaint, guidance_scale=guidance_scale_inpaint, strength=strength_inpaint, num_images_per_prompt=batch_count_inpaint).images

    # Convert generated images to PIL image format
    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]

    # Construct metadata string
    metadata_str = f"Seed: {seed}, Prompt: {prompt_i2i}, Negative Prompt: {negative_prompt_i2i}, Height: {height_inpaint}, Width: {width_inpaint}, Num Inference Steps: {num_inference_steps_inpaint}, Guidance Scale: {guidance_scale_inpaint}, Strength: {strength_inpaint}, Batch Count: {batch_count_inpaint}"

    return images_pil, images_pil, metadata_str
