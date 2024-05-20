css = """
                                                            /* Gradio Tweaks */
div.gradio-container {
  max-width: unset !important;
}
div.form {
  border-width: 0;
  box-shadow: none;
  background: transparent;
  overflow: visible;
  --block-background-fill: transparent;
}
.block.padded:not(.gradio-accordion) {
  padding-top: 1px !important;
  padding-bottom: 1px !important;
  padding-left: 0px !important;
  padding-right: 0px !important;
}
                                                              /* Global Model */                                                                  
.model_global {
  top: 0vh !important;
  width: 30vh !important;
  padding: 0 !important;
}

                                                                /* Txt2Img */

div.tab-nav.scroll-hide.svelte-kqij2n {
}
.txt2img_tab1 {
  padding-top: 0px !important;
  padding-bottom: 5px !important;
  padding-left: 8px !important;
  padding-right: 8px !important;
}
.prompt_t2i,
.negative_prompt_t2i {
  width: 75vw !important;
  height: 120px !important;
  z-index: 1 !important;
}
.generate_t2i {
  position: absolute !important;
  width: 20vw !important;
  height: 85px !important;
  left: 27.15vw !important;
  top: 3.7vh !important;
  background-image: linear-gradient(to bottom right, orange, red, yellow) !important;
}
.restore_faces_t2i {
  width: 0vw !important;
}
.scheduler_t2i,
.height_t2i,
.width_t2i,
.guidance_scale_t2i,
.seed_input_t2i {
  width: 22vw !important;
}
.num_inference_steps_t2i,
.batch_count_t2i,
.batch_size_t2i {
  width: 22vw !important;
  right: 8vw !important;
}
.image_output_t2i {
  width: 46vw !important;
  height: 45vh !important;
  right: 15vw !important;
} 
.metadata_t2i {
  width: 46vw !important;
  height: 17vh !important;
  right: 15vw !important;
}

                                                                /* Img2Img */


div.tab-nav.scroll-hide.svelte-kqij2n {
}
.img2img_tab1 {
  padding-top: 0px !important;
  padding-bottom: 5px !important;
  padding-left: 8px !important;
  padding-right: 8px !important;
}
.prompt_i2i,
.negative_prompt_i2i {
  width: 75vw !important;
  height: 120px !important;
  z-index: 1 !important;
}
.generate_i2i {
  position: absolute !important;
  width: 20vw !important;
  height: 85px !important;
  left: 11.15vw !important;
  bottom: 118vh !important;
  background-image: linear-gradient(to bottom right, orange, red, yellow) !important;
}
.image_input_i2i {
  width: 46vw !important;
  height: 45vh !important;
  left: 0vw !important;
}
.resize_mode_i2i {
  width: 33vw !important;
}
.restore_faces_i2i,
.hires_fix_i2i{
  width: 0vw !important;
}
.scheduler_i2i,
.height_i2i,
.width_i2i,
.guidance_scale_i2i,
.strength_i2i,
.seed_input_i2i {
  width: 22vw !important;
}
.num_inference_steps_i2i,
.batch_count_i2i,
.batch_size_i2i {
  width: 22vw !important;
  top: 54vh !important;
  right: 8vw !important;
}
.image_output_i2i {
  width: 46vw !important;
  height: 45vh !important;
  right: 15vw !important;
} 
.metadata_i2i {
  width: 46vw !important;
  height: 20vh !important;
  right: 15vw !important;
}

                                                                /* Inapaint */

div.tab-nav.scroll-hide.svelte-kqij2n {
}
.inpaint_tab1 {
  padding-top: 0px !important;
  padding-bottom: 5px !important;
  padding-left: 8px !important;
  padding-right: 8px !important;
}
.generate_inpaint {
  position: absolute !important;
  width: 20vw !important;
  height: 85px !important;
  left: 11.15vw !important;
  bottom: 145.9vh !important;
  background-image: linear-gradient(to bottom right, orange, red, yellow) !important;
}
.image_input_inpaint {
  width: 46vw !important;
  height: 45vh !important;
  left: 0vw !important;
}
.resize_mode_inpaint {
  width: 33vw !important;
}
.restore_faces_inpaint,
.hires_fix_inpaint{
  width: 0vw !important;
}
.scheduler_inpaint,
.height_inpaint,
.width_inpaint,
.guidance_scale_inpaint,
.strength_inpaint,
.seed_input_inpaint {
  width: 22vw !important;
}
.masked_padding_inpaint,
.num_inference_steps_inpaint,
.batch_count_inpaint,
.batch_size_inpaint {
  width: 22vw !important;
  top: 59.7vh !important;
  right: 8vw !important;
}
.image_output_inpaint {
  width: 46vw !important;
  height: 45vh !important;
  right: 15vw !important;
} 
.metadata_inpaint {
  width: 46vw !important;
  height: 20vh !important;
  right: 15vw !important;
}
"""
