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
from PIL import Image, ImageFilter, ImageOps

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

def crop_and_resize(image, width_i2i, height_i2i):
    """
    Crops the original image to match the aspect ratio of the target size, and then resizes the image.
    """
    # Get the original dimensions
    original_width, original_height = image.size

    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = width_i2i / height_i2i

    # Calculate new dimensions to crop
    if original_aspect_ratio > target_aspect_ratio:
        # Image is too wide, crop the width
        new_width = int(original_height * target_aspect_ratio)
        left = (original_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = original_height
    else:
        # Image is too tall, crop the height
        new_height = int(original_width / target_aspect_ratio)
        top = (original_height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = original_width

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Resize the cropped image to target dimensions
    resized_image = cropped_image.resize((width_i2i, height_i2i))

    return resized_image

def resize_and_fill(image, target_width, target_height):
    """
    Resizes the original image to fit the target width and height while maintaining its aspect ratio,
    and fills any extra space with blurred colors that match the original image's colors.
    """
    # Calculate the aspect ratios of the original image and target size
    original_width, original_height = image.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Resize the image while maintaining the aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        # Resize the image to the target width, calculate the new height
        new_width = target_width
        new_height = int(new_width / original_aspect_ratio)
    else:
        # Resize the image to the target height, calculate the new width
        new_height = target_height
        new_width = int(new_height * original_aspect_ratio)

    resized_image = image.resize((new_width, new_height))

    # Create a blank canvas with the target size
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))

    # Calculate the position to center the resized image on the canvas
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2

    # Paste the resized image onto the canvas
    canvas.paste(resized_image, (left, top))

    return canvas

def img2img(prompt_i2i, negative_prompt_i2i, image_input_i2i, resize_mode_i2i, height_i2i, width_i2i, num_inference_steps_i2i, guidance_scale_i2i, strength_i2i, batch_count_i2i, seed_int="", scheduler=None):
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

    # Validate pipeline
    global pipeline
    if pipeline is None:
        return "Pipeline is not loaded. Please click 'Load Pipeline' first."

    # Ensure height and width are divisible by 8
    height_i2i = closest_divisible_by_8(height_i2i)
    width_i2i = closest_divisible_by_8(width_i2i)

    # Convert the input NumPy array to a PIL image and convert to RGB format
    image = Image.fromarray(image_input_i2i).convert("RGB")

    # Resize mode handling
    if resize_mode_i2i == "Just Resize":
        # Resize the image
        image = image.resize((width_i2i, height_i2i))
    elif resize_mode_i2i == "Crop and Resize":
        # Crop and resize the image
        image = crop_and_resize(image, width_i2i, height_i2i)
    elif resize_mode_i2i == "Resize and Fill":
        # Resize and fill the image
        image = resize_and_fill(image, width_i2i, height_i2i)

    # Generate images using the pipeline
    images = pipe_module.pipeline(
        prompt=prompt_i2i,
        negative_prompt=negative_prompt_i2i,
        image=image,
        height=height_i2i,
        width=width_i2i,
        num_inference_steps=num_inference_steps_i2i,
        guidance_scale=guidance_scale_i2i,
        strength=strength_i2i,
        num_images_per_prompt=batch_count_i2i
    ).images

    # Convert generated images to PIL image format
    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]

    # Construct metadata string
    metadata_str = (
        f" Seed: {seed}, Prompt: {prompt_i2i}, Negative Prompt: {negative_prompt_i2i}, "
        f"Height: {height_i2i}, Width: {width_i2i}, Num Inference Steps: {num_inference_steps_i2i}, "
        f"Guidance Scale: {guidance_scale_i2i}, Strength: {strength_i2i}"
    )

    return images_pil, images_pil, metadata_str
