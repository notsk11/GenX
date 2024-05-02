# /content/modules/txt2img.py

import random
import sys
import numpy as np
from PIL import Image
import gradio as gr
import torch
from diffusers import DiffusionPipeline
from modules import pipeline
from modules import pipeline as pipe_module
from modules.pipeline import load_pipeline_txt2img

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

def txt2img(prompt_t2i, negative_prompt_t2i, height_t2i, width_t2i, num_inference_steps_t2i, guidance_scale_t2i, batch_count_t2i, seed_int="", scheduler=None):
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
    if pipe_module.pipeline is None:
        return "Pipeline is not loaded. Please click 'Load Pipeline' first."

    # Ensure height and width are divisible by 8
    height_t2i = closest_divisible_by_8(height_t2i)
    width_t2i = closest_divisible_by_8(width_t2i)

    images = pipe_module.pipeline(prompt=prompt_t2i, negative_prompt=negative_prompt_t2i, height=height_t2i, width=width_t2i, num_inference_steps=num_inference_steps_t2i, guidance_scale=guidance_scale_t2i, num_images_per_prompt=batch_count_t2i).images

    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]

    # Construct metadata string
    metadata_str = f" Seed: {seed}, Prompt: {prompt_t2i}, Negative Prompt: {negative_prompt_t2i}, Height: {height_t2i}, Width: {width_t2i}, Num Inference Steps: {num_inference_steps_t2i}, Guidance Scale: {guidance_scale_t2i}"
    return images_pil, images_pil, metadata_str
