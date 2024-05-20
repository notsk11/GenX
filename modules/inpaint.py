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
from modules.pipeline import load_model_onclick_inpaint
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

def crop_and_resize(image, width_inpaint, height_inpaint):
    """
    Crops the original image to match the aspect ratio of the target size, and then resizes the image.
    """
    # Get the original dimensions
    original_width, original_height = image.size

    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = width_inpaint / height_inpaint

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
    resized_image = cropped_image.resize((width_inpaint, height_inpaint))

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

    # If there is empty space on the canvas, fill it with a blurred version of the original image
    if new_width < target_width or new_height < target_height:
        # Resize the original image to the target size and apply a blur filter
        blurred_original = image.resize((target_width, target_height)).filter(ImageFilter.GaussianBlur(radius=20))

        # Blend the blurred original with the canvas where there are transparent pixels
        # Create an alpha mask to blend the blurred image with the canvas
        alpha_mask = Image.new("L", (target_width, target_height), 255)
        mask = ImageOps.invert(alpha_mask)

        # Use alpha_composite to blend the images
        canvas = Image.composite(blurred_original, canvas, mask)
        
    return canvas

def inpaint(model_id, prompt_i2i, negative_prompt_i2i, image_input_inpaint, resize_mode_inpaint, mask_blur_inpaint, mask_mode_inpaint, masked_padding_inpaint, height_inpaint, width_inpaint, num_inference_steps_inpaint, guidance_scale_inpaint, strength_inpaint, batch_count_inpaint, seed_int="", scheduler=None):
    load_model_onclick_inpaint(model_id)
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


    # Convert `batch_count_inpaint` to integer and handle empty string
    if batch_count_inpaint == "":
        batch_count_inpaint = 1  # Set a default value if empty
    else:
        try:
            batch_count_inpaint = int(batch_count_inpaint)
        except ValueError:
            print("Invalid input for batch count. Please enter a valid number.")
            return None

    if isinstance(image_input_inpaint, dict):
        image = image_input_inpaint.get('image').convert("RGB")
        mask_img = image_input_inpaint.get('mask').convert("L")  # Convert mask to grayscale
        # Resize mode handling
        if resize_mode_inpaint == "Just Resize":
            # Resize the image and mask
            image = image.resize((width_inpaint, height_inpaint))
            mask_img = mask_img.resize((width_inpaint, height_inpaint))
        # Check resize mode
        if resize_mode_inpaint == "Crop and Resize":
            # Crop and resize the image and mask
            image = crop_and_resize(image, width_inpaint, height_inpaint)
            mask_img = crop_and_resize(mask_img, width_inpaint, height_inpaint)
        # Check resize mode
        if resize_mode_inpaint == "Resize and Fill":
            # Resize and fill the image and mask
            image = resize_and_fill(image, width_inpaint, height_inpaint)
            mask_img = resize_and_fill(mask_img, width_inpaint, height_inpaint)
        # Conditional inversion of the mask based on the selected mask mode
        if mask_mode_inpaint == "Inpaint Not Masked":
            # Invert the grayscale mask
            mask_img = Image.eval(mask_img, lambda x: 255 - x)

        # Blur the mask if needed
        mask_img_blur = pipe_module.pipeline.mask_processor.blur(mask_img, blur_factor=mask_blur_inpaint)
    else:
        image = image_input_inpaint.convert("RGB")
        mask_img_blur = None

    height_inpaint = closest_divisible_by_8(height_inpaint)
    width_inpaint = closest_divisible_by_8(width_inpaint)
    # Generate images using the pipeline
    images = pipe_module.pipeline(prompt=prompt_i2i, negative_prompt=negative_prompt_i2i, image=image, mask_image=mask_img_blur, padding_mask_crop=masked_padding_inpaint, height=height_inpaint, width=width_inpaint, num_inference_steps=num_inference_steps_inpaint, guidance_scale=guidance_scale_inpaint, strength=strength_inpaint, num_images_per_prompt=batch_count_inpaint).images

    # Convert generated images to PIL image format
    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]

    # Construct metadata string
    metadata_str = f"Seed: {seed}, Prompt: {prompt_i2i}, Negative Prompt: {negative_prompt_i2i}, Height: {height_inpaint}, Width: {width_inpaint}, Num Inference Steps: {num_inference_steps_inpaint}, Guidance Scale: {guidance_scale_inpaint}, Strength: {strength_inpaint}, Batch Count: {batch_count_inpaint}"

    return images_pil, metadata_str
