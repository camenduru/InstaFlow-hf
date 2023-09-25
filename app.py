import gradio as gr

from rf_models import RF_model
from sd_models import SD_model

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

from diffusers import StableDiffusionXLImg2ImgPipeline
import time
import copy
import numpy as np

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

global model
global base_model
global img

def set_model(model_id):
    global model 
    if model_id == "InstaFlow-0.9B":
        model = RF_model("./instaflow_09b.pt")
    elif model_id == "InstaFlow-1.7B":
        model = RF_model("./instaflow_17b.pt")
    else:
        raise NotImplementedError
    print('Finished Loading Model!')

def set_base_model(model_id):
    global base_model 
    if model_id == "runwayml/stable-diffusion-v1-5":
        base_model = SD_model("runwayml/stable-diffusion-v1-5")
    else:
        raise NotImplementedError
    print('Finished Loading Base Model!')

def set_new_latent_and_generate_new_image(seed, prompt, num_inference_steps=1, guidance_scale=0.0):
    print('Generate with input seed')
    global model
    global img
    negative_prompt=""
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    print(seed, num_inference_steps, guidance_scale)

    t_s = time.time()
    new_image = model.set_new_latent_and_generate_new_image(int(seed), prompt, negative_prompt, int(num_inference_steps), guidance_scale)
    inf_time = time.time() - t_s 

    img = copy.copy(new_image[0])

    return new_image[0], inf_time

def set_new_latent_and_generate_new_image_with_base_model(seed, prompt, num_inference_steps=1, guidance_scale=0.0):
    print('Generate with input seed')
    global base_model
    negative_prompt=""
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    print(seed, num_inference_steps, guidance_scale)

    t_s = time.time()
    new_image = base_model.set_new_latent_and_generate_new_image(int(seed), prompt, negative_prompt, int(num_inference_steps), guidance_scale)
    inf_time = time.time() - t_s

    return new_image[0], inf_time


def set_new_latent_and_generate_new_image_and_random_seed(seed, prompt, negative_prompt="", num_inference_steps=1, guidance_scale=0.0):
    print('Generate with a random seed')
    global model
    global img
    seed = np.random.randint(0, 2**32)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    print(seed, num_inference_steps, guidance_scale)

    t_s = time.time()
    new_image = model.set_new_latent_and_generate_new_image(int(seed), prompt, negative_prompt, int(num_inference_steps), guidance_scale)
    inf_time = time.time() - t_s

    img = copy.copy(new_image[0])

    return new_image[0], seed, inf_time


def refine_image_512(prompt):
    print('Refine with SDXL-Refiner (512)')
    global img

    t_s = time.time()
    img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
    img = img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    new_image = pipe(prompt, image=img).images[0] 
    print('time consumption:', time.time() - t_s) 
    new_image = np.array(new_image) * 1.0 / 255.

    img = new_image

    return new_image

def refine_image_1024(prompt):
    print('Refine with SDXL-Refiner (1024)')
    global img

    t_s = time.time()
    img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
    img = torch.nn.functional.interpolate(img, size=1024, mode='bilinear')
    img = img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    new_image = pipe(prompt, image=img).images[0] 
    print('time consumption:', time.time() - t_s) 
    new_image = np.array(new_image) * 1.0 / 255.

    img = new_image

    return new_image

set_model('InstaFlow-0.9B')
set_base_model("runwayml/stable-diffusion-v1-5")

with gr.Blocks() as gradio_gui:
    gr.Markdown(
    """
    # InstaFlow! One-Step Stable Diffusion with Rectified Flow
    ## This Huggingface Space provides a demo of one-step InstaFlow-0.9B and measures the inference time. 
    ## For fair comparison, Stable Difusion 1.5 is shown in parallel.
    ## 
    """)
    gr.Markdown("Set Input Seed and Text Prompts Here")
    with gr.Row():
        with gr.Column(scale=0.4):
            seed_input = gr.Textbox(value='101098274', label="Random Seed") 
        with gr.Column(scale=0.4):
            prompt_input = gr.Textbox(value='A high-resolution photograph of a waterfall in autumn; muted tone', label="Prompt")

    with gr.Row():
        with gr.Column(scale=0.4):
            with gr.Group():
                gr.Markdown("Generation from InstaFlow-0.9B")
                im = gr.Image()
            
            gr.Markdown("Model ID: One-Step InstaFlow-0.9B")
            inference_time_output = gr.Textbox(value='0.0', label='Inference Time with One-Step Model (Second)')
            num_inference_steps = gr.Textbox(value='1', label="Number of Inference Steps (can only be 1)")
            guidance_scale = gr.Textbox(value='0.0', label="Guidance Scale for InstaFlow (can only be 0.0)")

            new_image_button = gr.Button(value="One-Step Generation with InstaFlow and the Input Seed")
            new_image_button.click(set_new_latent_and_generate_new_image, inputs=[seed_input, prompt_input, num_inference_steps, guidance_scale], outputs=[im, inference_time_output])

            refine_button_512 = gr.Button(value="Refine One-Step Generation with SDXL Refiner (Resolution: 512)")
            refine_button_512.click(refine_image_512, inputs=[prompt_input], outputs=[im])

        with gr.Column(scale=0.4):
            with gr.Group():
                gr.Markdown("Generation from Stable Diffusion 1.5") 
                im_base = gr.Image()

            gr.Markdown("Model ID: Multi-Step Stable Diffusion 1.5")
            base_model_inference_time_output = gr.Textbox(value='0.0', label='Inference Time with Multi-Step Stable Diffusion (Second)')

            base_num_inference_steps = gr.Textbox(value='25', label="Number of Inference Steps for Stable Diffusion")
            base_guidance_scale = gr.Textbox(value='5.0', label="Guidance Scale for Stable Diffusion")
            
            base_new_image_button = gr.Button(value="Multi-Step Generation with Stable Diffusion and the Input Seed") 
            base_new_image_button.click(set_new_latent_and_generate_new_image_with_base_model, inputs=[seed_input, prompt_input,  base_num_inference_steps, base_guidance_scale], outputs=[im_base, base_model_inference_time_output])

gradio_gui.launch()   
