# /content/app.py

import gradio as gr
from modules import style, txt2img, img2img, inpaint, pipeline
from modules.style import css
from modules.txt2img import txt2img
from modules.img2img import img2img
from modules.inpaint import inpaint
from modules.pipeline import load_pipeline_txt2img
from modules.pipeline import load_pipeline_img2img
from modules.pipeline import load_pipeline_inpaint
from modules.pipeline import update_scheduler



with gr.Blocks(css=css) as demo:
  with gr.Column():
    with gr.Row():
      with gr.Column():
        model_global = gr.Textbox(label="Stable Diffusion Checkpoint", value="SG161222/Realistic_Vision_V6.0_B1_noVAE", elem_classes="model_global")
  with gr.Tab("Txt2Img", elem_classes="txt2img_tab1"):
    with gr.Column():
      with gr.Row():
        with gr.Column():
          prompt_t2i = gr.Textbox(lines=3, show_label=False, placeholder="Place Prompts here...", elem_classes="prompt_t2i")
          negative_prompt_t2i = gr.Textbox(lines=3, show_label=False, placeholder="Place Negative Prompts here...", elem_classes="negative_prompt_t2i")
        with gr.Column():
          generate_t2i = gr.Button(value="Generate", elem_classes="generate_t2i")
      with gr.Tab("Generation", elem_classes="txt2img_tab2"):
        with gr.Row():
            with gr.Column():
              scheduler_t2i = gr.Dropdown(elem_classes="scheduler_t2i", label="Sampling Methods", choices=[
              "PNDM",
              "DEIS",
              "UniPC",
              "Euler",
              "Euler-A",
              "LMS",
              "LMS-Karras",
              "DPM2",
              "DPM2-Karras",
              "DPM-SDE",
              "DPM-SDE-Karras",
              "DPM-2M",
              "DPM-2M-Karras",
              "DPM2-A",
              "DPM2-A-Karras",
              "DPM-2M-SDE",
              "DPM-2M-SDE-Karras"], value="DPM-SDE-Karras")
              fix_t2i = gr.CheckboxGroup(choices=["Restore Faces", "Hires Fix"], container=False, elem_classes="fix_t2i")
              height_t2i = gr.Slider(elem_classes="height_t2i", label="Height", minimum=100, maximum=1600, value=408, step=4)
              width_t2i = gr.Slider(elem_classes="width_t2i", label="Width", minimum=100, maximum=1600, value=408, step=4)
              guidance_scale_t2i = gr.Slider(elem_classes="guidance_scale_t2i", label="CFG Scale", minimum=0, maximum=10, value=4, step=0.1)
              seed_input_t2i = gr.Textbox(elem_classes="seed_input_t2i", label="Seed")
            with gr.Column():
              num_inference_steps_t2i = gr.Slider(elem_classes="num_inference_steps_t2i", label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)
              batch_count_t2i = gr.Slider(elem_classes="batch_count_t2i", label="Batch Count", minimum=1, maximum=10, step=1, value=1)
              batch_size_t2i = gr.Slider(elem_classes="batch_size_t2i", label="Batch Size", minimum=1, maximum=10, value=1, step=1)
            with gr.Column():
              image_output_t2i = gr.Gallery(elem_classes="image_output_t2i")
              metadata_t2i = gr.Textbox(elem_classes="metadata_t2i", label="Metadata", lines=10, show_copy_button=True)
              load_model_t2i = gr.Button(value="Load", elem_classes="load_model_global")
    with gr.Accordion(elem_classes="zoom_t2i_accordion", label="Zoomed", open=False):
      image_output_zoom_t2i = gr.Gallery(elem_classes="image_output_zoomed_t2i", container=True, interactive=True)
      load_model_t2i.click(fn=load_pipeline_txt2img, inputs=[model_global])
      scheduler_t2i.change(fn=update_scheduler, inputs=[scheduler_t2i])
      generate_t2i.click(fn=txt2img, inputs=[prompt_t2i, negative_prompt_t2i, height_t2i, width_t2i, num_inference_steps_t2i, guidance_scale_t2i, batch_count_t2i, seed_input_t2i], outputs=[image_output_t2i, image_output_zoom_t2i, metadata_t2i])


  with gr.Tab("Img2Img", elem_classes="img2img_tab1"):
    with gr.Column():
      with gr.Row():
        with gr.Column():
          prompt_i2i = gr.Textbox(lines=3, show_label=False, placeholder="Place Prompts here...", elem_classes="prompt_i2i")
          negative_prompt_i2i = gr.Textbox(lines=3, show_label=False, placeholder="Place Negative Prompts here...", elem_classes="negative_prompt_i2i")
      with gr.Tab("Img2Img", elem_classes="img2img_tab2"):
        with gr.Row():
            with gr.Column():
              image_input_i2i = gr.Image(elem_classes="image_input_i2i")
              resize_mode_i2i = gr.CheckboxGroup(value="Just Resize", label="Resize Mode", elem_classes="resize_mode_i2i", choices=["Just Resize", "Crop and Resize", "Resize and Fill", "Just Resize(Latent Upscale)"])
              scheduler_i2i = gr.Dropdown(elem_classes="scheduler_i2i", label="Sampling Methods", choices=[
              "PNDM",
              "DEIS",
              "UniPC",
              "Euler",
              "Euler-A",
              "LMS",
              "LMS-Karras",
              "DPM2",
              "DPM2-Karras",
              "DPM-SDE",
              "DPM-SDE-Karras",
              "DPM-2M",
              "DPM-2M-Karras",
              "DPM2-A",
              "DPM2-A-Karras",
              "DPM-2M-SDE",
              "DPM-2M-SDE-Karras"], value="DPM-SDE-Karras")
              fix_i2i = gr.CheckboxGroup(choices=["Restore Faces", "Hires Fix"], container=False, elem_classes="fix_i2i")
              height_i2i = gr.Slider(elem_classes="height_i2i", label="Height", minimum=100, maximum=1600, value=408, step=4)
              width_i2i = gr.Slider(elem_classes="width_i2i", label="Width", minimum=100, maximum=1600, value=408, step=4)
              guidance_scale_i2i = gr.Slider(elem_classes="guidance_scale_i2i", label="CFG Scale", minimum=0, maximum=10, value=4, step=0.1)
              strength_i2i = gr.Slider(elem_classes="strength_i2i", label="Denoising Strength", minimum=0, maximum=1, value=0.4, step=0.01)
              seed_input_i2i = gr.Textbox(elem_classes="seed_input_i2i", label="Seed")
            with gr.Column():
              num_inference_steps_i2i = gr.Slider(elem_classes="num_inference_steps_i2i", label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)
              batch_count_i2i = gr.Slider(elem_classes="batch_count_i2i", label="Batch Count", minimum=1, maximum=10, step=1, value=1)
              batch_size_i2i = gr.Slider(elem_classes="batch_size_i2i", label="Batch Size", minimum=1, maximum=10, value=1, step=1)
            with gr.Column():
              image_output_i2i = gr.Gallery(elem_classes="image_output_i2i")
              metadata_i2i = gr.Textbox(elem_classes="metadata_i2i", label="Metadata", lines=10, show_copy_button=True)
              load_model_i2i = gr.Button(value="Load", elem_classes="load_model_global")
              generate_i2i = gr.Button(value="Generate", elem_classes="generate_i2i")
        with gr.Accordion(elem_classes="zoom_i2i_accordion", label="Zoomed", open=False):
          image_output_zoom_i2i = gr.Gallery(elem_classes="image_output_zoomed_i2i", container=True, interactive=True)
          load_model_i2i.click(fn=load_pipeline_img2img, inputs=[model_global])
          scheduler_i2i.change(fn=update_scheduler, inputs=[scheduler_i2i])
          generate_i2i.click(fn=img2img, inputs=[prompt_i2i, negative_prompt_i2i, height_i2i, width_i2i, num_inference_steps_i2i, guidance_scale_i2i, strength_i2i, batch_count_i2i, seed_input_i2i], outputs=[image_output_i2i, image_output_zoom_i2i, metadata_i2i])


      with gr.Tab("Inpaint", elem_classes="img2img_tab3"):
        with gr.Row():
            with gr.Column():
              image_input_inpaint = gr.Image(elem_classes="image_input_inpaint", tool='sketch', type='pil')
              resize_mode_inpaint = gr.CheckboxGroup(value="Just Resize", label="Resize Mode", elem_classes="resize_mode_inpaint", choices=["Just Resize", "Crop and Resize", "Resize and Fill", "Just Resize(Latent Upscale)"])
              mask_blur_inpaint = gr.Slider(elem_classes="mask_blur_inpaint", label="Mask Blur", minimum=0, maximum=50, value=4, step=1)
              mask_mode_inpaint = gr.CheckboxGroup(elem_classes="mask_mode_inpaint", label="Mask Mode", choices=["Inpaint Masked", "Inpaint Not Masked"], value="Inpaint Masked")
              marked_content_inpaint = gr.CheckboxGroup(elem_classes="marked_content_inpaint", label="Marked Content", choices=["Fill", "Original", "Latent Noise", "Latent Nothing"], value="Original")
              inpaint_area_inpaint = gr.CheckboxGroup(elem_classes="inpaint_area_inpaint", label="Inpaint Area", choices=["Whole Area", "Only Masked"], value="Only Masked")
              scheduler_inpaint = gr.Dropdown(elem_classes="scheduler_inpaint", label="Sampling Methods", choices=[
              "PNDM",
              "DEIS",
              "UniPC",
              "Euler",
              "Euler-A",
              "LMS",
              "LMS-Karras",
              "DPM2",
              "DPM2-Karras",
              "DPM-SDE",
              "DPM-SDE-Karras",
              "DPM-2M",
              "DPM-2M-Karras",
              "DPM2-A",
              "DPM2-A-Karras",
              "DPM-2M-SDE",
              "DPM-2M-SDE-Karras"], value="DPM-SDE-Karras")
              fix_inpaint = gr.CheckboxGroup(choices=["Restore Faces", "Hires Fix"], container=False, elem_classes="fix_inpaint")
              height_inpaint = gr.Slider(elem_classes="height_inpaint", label="Height", minimum=100, maximum=1600, value=408, step=4)
              width_inpaint = gr.Slider(elem_classes="width_inpaint", label="Width", minimum=100, maximum=1600, value=408, step=4)
              guidance_scale_inpaint = gr.Slider(elem_classes="guidance_scale_inpaint", label="CFG Scale", minimum=0, maximum=10, value=4, step=0.1)
              strength_inpaint = gr.Slider(elem_classes="strength_inpaint", label="Denoising Strength", minimum=0, maximum=1, value=0.4, step=0.01)
              seed_input_inpaint = gr.Textbox(elem_classes="seed_input_inpaint", label="Seed")
            with gr.Column():
              masked_padding_inpaint = gr.Slider(elem_classes="masked_padding_inpaint", label="Only Masked Padding, Pixels", minimum=1, maximum=200, value=32, step=1)
              num_inference_steps_inpaint = gr.Slider(elem_classes="num_inference_steps_inpaint", label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)
              batch_count_inpaint = gr.Slider(elem_classes="batch_count_inpaint", label="Batch Count", minimum=1, maximum=10, step=1, value=1)
              batch_size_inpaint = gr.Slider(elem_classes="batch_size_inpaint", label="Batch Size", minimum=1, maximum=10, value=1, step=1)
            with gr.Column():
              image_output_inpaint = gr.Gallery(elem_classes="image_output_inpaint")
              metadata_inpaint = gr.Textbox(elem_classes="metadata_inpaint", label="Metadata", lines=10, show_copy_button=True)
              load_model_inpaint = gr.Button(value="Load", elem_classes="load_model_global")
              generate_inpaint = gr.Button(value="Generate", elem_classes="generate_inpaint")
        with gr.Accordion(elem_classes="zoom_inpaint_accordion", label="Zoomed", open=False):
          image_output_zoom_inpaint = gr.Gallery(elem_classes="image_output_zoomed_inpaint", container=True, interactive=True)
          load_model_inpaint.click(fn=load_pipeline_inpaint, inputs=[model_global])
          scheduler_inpaint.change(fn=update_scheduler, inputs=[scheduler_inpaint])
          generate_inpaint.click(fn=inpaint, inputs=[prompt_i2i, negative_prompt_i2i, image_input_inpaint, height_inpaint, width_inpaint, num_inference_steps_inpaint, guidance_scale_inpaint, strength_inpaint, batch_count_inpaint, seed_input_inpaint], outputs=[image_output_inpaint, image_output_zoom_inpaint, metadata_inpaint])


demo.launch(share=True, debug=True)
