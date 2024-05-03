css = """
/* SETUP CONTAINER AND BASICS */

div.gradio-container {
  max-width: unset !important;
}
:root, .dark {
    --checkbox-label-gap: 0.25em 0.1em;
    --section-header-text-size: 12pt;
    --block-background-fill: transparent;
}
div.form {
    border-width: 0;
    box-shadow: none;
    background: transparent;
    overflow: visible;
}
.block.padded:not(.gradio-accordion) {
  padding-top: 1px !important;
  padding-bottom: 1px !important;
  padding-left: 0px !important;
  padding-right: 0px !important;
}

/* SETUP COMPUNENTS */

  /* GLOBAL MODEL */

.model_global {
  top: -10px;
  width: 300px !important;
  padding: 0 !important;
}


 /* TXT 2 IMAGE */


.txt2img_tab1 {
  padding: 8px !important;
}
.load_model_global {
  left: -945px !important;
  top: -355px !important;
  width: 70px !important;
  position: absolute !important;
  background-image: conic-gradient(red, yellow, red, yellow, red) !important;
} 
.prompt_t2i {
  width: 1450px !important;
  height: 95px !important;
}
.negative_prompt_t2i {
  width: 1450px !important;
  height: 90px !important;
}
.generate_t2i {
  left: 555px;
  width: 330px !important;
  height: 85px !important;
  position: absolute;
  background-image: linear-gradient(to bottom right, orange, red, yellow) !important;
}
.txt2img_tab2 {
  height: 900px !important;
}

    /* 1ST COLUMN */

.scheduler_t2i {
  width: 440px !important;
  height: 70px !important;
}
.fix_t2i {
  top: -5px !important;
}
.height_t2i {
  top: -15px !important;
  width: 540px !important;
}
.width_t2i {
  top: -15px !important;
  width: 540px !important;
}
.guidance_scale_t2i {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}
.seed_input_t2i {
  top: -15px !important;
  width: 915px !important; 
}

    /* 2ND COLUMN */

.num_inference_steps_t2i {
  left: -140px !important;
  width: 440px !important;
}
.batch_count_t2i {
  left: -40px !important;
  top: 70px !important;
  width: 340px !important;
}
.batch_size_t2i {
  left: -40px !important;
  top: 70px !important;
  width: 340px !important;
}

    /* 3RD COLUMN */

.image_output_t2i {
  width: 900px !important;
  height: 500px !important;
  left: -300px !important;
}
.metadata_t2i {
  width: 900px !important;
  left: -300px !important;
  top: -10px !important;
  border: none !important;
}

  /* IMAGE ZOOMED OUTPUT */

.zoom_t2i_accordion {
  width: 130px !important;
  height: 30px !important;
  top: -1280px !important;
  left: 1730px !important;
}
.image_output_zoomed_t2i {
  width: 1870px !important;
  height: 835px !important;
  top: 35px !important;
  left: -1740px !important;
  object-fit: cover !important;
}
.image_output_zoomed_t2i img{
    object-fit: contain !important;
}


 /* IMAGE 2 IMAGE */


.img2img_tab1 {
  padding: 8px !important;
} 
.prompt_i2i {
  width: 1450px !important;
  height: 95px !important;
}
.negative_prompt_i2i {
  width: 1450px !important;
  height: 90px !important;
}
.generate_i2i {
  top: -245px !important;
  left: 248px !important;
  width: 330px !important;
  height: 85px !important;
  position: absolute;
  background-image: linear-gradient(to bottom right, orange, red, yellow) !important;
}
.img2img_tab2 {
  height: auto !important;
}

    /* 1ST COLUMN  I2I*/
.image_input_i2i {
  width: 900px !important;
  height: 500px !important;
}
.resize_mode_i2i {
  width: 700px !important;
}
.scheduler_i2i {
  width: 440px !important;
  height: 70px !important;
}
.fix_i2i {
  top: -5px !important;
}
.height_i2i {
  top: -15px !important;
  width: 540px !important;
}
.width_i2i {
  top: -15px !important;
  width: 540px !important;
}
.guidance_scale_i2i {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}
.strength_i2i {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}
.seed_input_i2i {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}

    /* 2ND COLUMN I2I */

.num_inference_steps_i2i {
  left: -140px !important;
  top: 585px !important;
  width: 440px !important;
}
.batch_count_i2i {
  left: -40px !important;
  top: 655px !important;
  width: 340px !important;
}
.batch_size_i2i {
  left: -40px !important;
  top: 655px !important;
  width: 340px !important;
}

    /* 3RD COLUMN I2I */

.image_output_i2i {
  width: 900px !important;
  height: 500px !important;
  left: -300px !important;
}
.metadata_i2i {
  width: 900px !important;
  left: -300px !important;
  top: -10px !important;
  border: none !important;
}

  /* IMAGE ZOOMED OUTPUT IMAGE2IMAGE */

.zoom_i2i_accordion {
  width: 130px !important;
  height: 30px !important;
  top: -1383px !important;
  left: 1717px !important;
}
.image_output_zoomed_i2i {
  width: 1870px !important;
  height: 835px !important;
  top: 35px !important;
  left: -1740px !important;
  object-fit: cover !important;
}
.image_output_zoomed_i2i img {
    object-fit: contain !important;
}


 /* INPAINT */


    /* 1ST COLUMN INPAINT */


.image_input_inpaint {
  width: 900px !important;
  height: 500px !important;
}
.resize_mode_inpaint {
  width: 700px !important;
}
.mask_blur_inpaint {
  width: 912px !important;
}
.mask_mode_inpaint {
  width: 700px !important;
}
.marked_content_inpaint {
  width: 700px !important;
}
.inpaint_area_inpaint {
  width: 300px !important; 
}
.scheduler_inpaint {
  width: 440px !important;
  height: 70px !important; 
}
.fix_inpaint {
  top: -5px !important;
}
.height_inpaint {
  top: -15px !important;
  width: 540px !important;
}
.width_inpaint {
  top: -15px !important;
  width: 540px !important;
}
.guidance_scale_inpaint {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}
.strength_inpaint {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}
.seed_input_inpaint {
  top: -15px !important;
  width: 915px !important;
  z-index: 1 !important;
}

    /* 2ND COLUMN INPAINT */
.masked_padding_inpaint {
  left: -310px !important;
  top: 765px !important;
  width: 610px !important;
}
.num_inference_steps_inpaint {
  left: -140px !important;
  top: 780px !important;
  width: 440px !important;
}
.batch_count_inpaint {
  left: -40px !important;
  top: 855px !important;
  width: 340px !important;
}
.batch_size_inpaint {
  left: -40px !important;
  top: 855px !important;
  width: 340px !important;
}

    /* 3RD COLUMN INPAINT */

.image_output_inpaint {
  width: 900px !important;
  height: 500px !important;
  left: -300px !important;
}
.metadata_inpaint {
  width: 900px !important;
  left: -300px !important;
  top: -10px !important;
  border: none !important;
}
.generate_inpaint {
  top: -245px !important;
  left: 248px !important;
  width: 330px !important;
  height: 85px !important;
  position: absolute;
  background-image: linear-gradient(to bottom right, orange, red, yellow) !important;
}
  /* IMAGE ZOOMED OUTPUT INPAINT */

.zoom_inpaint_accordion {
  width: 130px !important;
  height: 30px !important;
  top: -1630px !important;
  left: 1717px !important;
}
.image_output_zoomed_inpaint {
  width: 1870px !important;
  height: 835px !important;
  top: 35px !important;
  left: -1740px !important;
  object-fit: cover !important;
}
.image_output_zoomed_inpaint img {
    object-fit: contain !important;
}
"""
