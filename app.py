import gradio as gr

from rf_models import RF_model

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


def set_new_latent_and_generate_new_image(seed, prompt, negative_prompt="", num_inference_steps=1, guidance_scale=0.0):
    print('Generate with input seed')
    global model
    global img
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    print(seed, num_inference_steps, guidance_scale)

    t_s = time.time()
    new_image = model.set_new_latent_and_generate_new_image(int(seed), prompt, negative_prompt, int(num_inference_steps), guidance_scale)
    print('time consumption:', time.time() - t_s) 

    img = copy.copy(new_image[0])

    return new_image[0]

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
    print('time consumption:', time.time() - t_s) 

    img = copy.copy(new_image[0])

    return new_image[0], seed


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

with gr.Blocks() as gradio_gui:

    with gr.Row():
        with gr.Column(scale=0.5):
            im = gr.Image()

        with gr.Column():
            #model_id = gr.Dropdown(["InstaFlow-0.9B", "InstaFlow-1.7B"], label="Model ID", info="Choose Your Model")
 
            #set_model_button = gr.Button(value="Set New Model")
            #set_model_button.click(set_model, inputs=[model_id])

            model_id = gr.Textbox(value='InstaFlow-0.9B', label="Model ID")

            seed_input = gr.Textbox(value='101098274', label="Random Seed")
            prompt_input = gr.Textbox(value='A high-resolution photograph of a waterfall in autumn; muted tone', label="Prompt")
             
            new_image_button = gr.Button(value="Generate Image with the Input Seed")
            new_image_button.click(set_new_latent_and_generate_new_image, inputs=[seed_input, prompt_input], outputs=[im])

            next_image_button = gr.Button(value="Generate Image with a Random Seed")
            next_image_button.click(set_new_latent_and_generate_new_image_and_random_seed, inputs=[seed_input, prompt_input], outputs=[im, seed_input])
            

            refine_button_512 = gr.Button(value="Refine with Refiner (Resolution: 512)")
            refine_button_512.click(refine_image_512, inputs=[prompt_input], outputs=[im])

            refine_button_1024 = gr.Button(value="Refine with Refiner (Resolution: 1024)")
            refine_button_1024.click(refine_image_1024, inputs=[prompt_input], outputs=[im])


gradio_gui.launch()   
