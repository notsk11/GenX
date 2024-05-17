# /content/modules/pipeline.py

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
from diffusers import (
    PNDMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
)

pipeline = None

def pipeline_main_load(model_id):
    pipeline_obj = AutoPipelineForText2Image.from_pretrained(model_id)
    pipeline_obj.safety_checker = None
    return pipeline_obj

# Txt2Img Pipeline
def load_model_onclick_t2i(model_id):
    global pipeline
    if pipeline is None:
        pipeline = pipeline_main_load(model_id)
        pipeline = load_pipeline_txt2img(model_id)
    else:
        pipeline = load_pipeline_txt2img(model_id)

def load_pipeline_txt2img(model_id):
  global pipeline
  pipeline = AutoPipelineForText2Image.from_pipe(pipeline).to('cuda')
  return pipeline

# Img2Img Pipeline
def load_model_onclick_i2i(model_id):
    global pipeline
    if pipeline is None:
        pipeline = pipeline_main_load(model_id)
        pipeline = load_pipeline_img2img(model_id)
    else:
        pipeline = load_pipeline_img2img(model_id)

def load_pipeline_img2img(model_id):
  global pipeline
  pipeline = AutoPipelineForImage2Image.from_pipe(pipeline).to('cuda')
  return pipeline

# Inpaint Pipeline
def load_model_onclick_inpaint(model_id):
    global pipeline
    if pipeline is None:
        pipeline = pipeline_main_load(model_id)
        pipeline = load_pipeline_inpaint(model_id)
    else:
        pipeline = load_pipeline_inpaint(model_id)

def load_pipeline_inpaint(model_id):
  global pipeline
  pipeline = AutoPipelineForInpainting.from_pipe(pipeline).to('cuda')
  return pipeline

# Schedulers
def update_scheduler(scheduler):
    if scheduler == "PNDM":
        input_scheduler = PNDMScheduler.from_pretrained("notsk007/PNDM")
    elif scheduler == "DEIS":
        input_scheduler = DEISMultistepScheduler.from_pretrained("notsk007/DEIS")
    elif scheduler == "UniPC":
        input_scheduler = UniPCMultistepScheduler.from_pretrained("notsk007/UniPC")
    elif scheduler == "Euler":
        input_scheduler = EulerDiscreteScheduler.from_pretrained("notsk007/Euler")
    elif scheduler == "Euler-A":
        input_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("notsk007/Euler-A")
    elif scheduler == "LMS":
        input_scheduler = LMSDiscreteScheduler.from_pretrained("notsk007/LMS")
    elif scheduler == "LMS-Karras":
        input_scheduler = LMSDiscreteScheduler.from_pretrained("notsk007/LMS-Karras")
    elif scheduler == "DPM2":
        input_scheduler = KDPM2DiscreteScheduler.from_pretrained("notsk007/DPM2")
    elif scheduler == "DPM2-Karras":
        input_scheduler = KDPM2DiscreteScheduler.from_pretrained("notsk007/DPM2-Karras")
    elif scheduler == "DPM2-A":
        input_scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained("notsk007/DPM2-A")
    elif scheduler == "DPM2-A-Karras":
        input_scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained("notsk007/DPM2-A-Karras")
    elif scheduler == "DPM-SDE":
        input_scheduler = DPMSolverSinglestepScheduler.from_pretrained("notsk007/DPM-SDE")
    elif scheduler == "DPM-SDE-Karras":
        input_scheduler = DPMSolverSinglestepScheduler.from_pretrained("notsk007/DPM-SDE-Karras")
    elif scheduler == "DPM-2M":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M")
    elif scheduler == "DPM-2M-Karras":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M-Karras")
    elif scheduler == "DPM-2M-SDE":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M-SDE")
    elif scheduler == "DPM-2M-SDE-Karras":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M-SDE-Karras")
    else:
        print("Invalid scheduler selection.")
        return

    pipeline.scheduler = input_scheduler
    print(f"Scheduler updated to: {scheduler}")
    return pipeline.scheduler
