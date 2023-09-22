#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional, Union, List, Callable

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel#, StackUNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

import time

from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision

import cv2        

def inference_latent(
    pipeline,
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
):
    
    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    #pipeline.check_inputs(prompt, height, width, callback_steps)

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = pipeline._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    #setup_seed(0)
    text_embeddings = pipeline._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.in_channels
    latents = latents.reshape(1, num_channels_latents, 64, 64)

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - \
        num_inference_steps * pipeline.scheduler.order

    latents_cllt = [latents.detach().clone()]
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(
                latent_model_input, t)

            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            outputs = pipeline.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs)
            
            latents = outputs.prev_sample
            

    example = {
        'latent': latents.detach().clone(),
        'text_embeddings': text_embeddings.chunk(2)[1].detach() if do_classifier_free_guidance else text_embeddings.detach(),
    }
    return example



def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()


class SD_model():

    def __init__(self, pretrained_model_name_or_path):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer"#, revision=args.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder"#, revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae"#, revision=args.revision
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet"#, revision=args.non_ema_revision
        )


        unet.eval()
        vae.eval()
        text_encoder.eval()

        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float16
        self.weight_dtype = weight_dtype
        device = 'cuda'
        self.device = device

        # Move text_encode and vae to gpu and cast to weight_dtype
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)

        # Create the pipeline using the trained modules and save it.
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(device) 
        from diffusers import DPMSolverMultistepScheduler

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        self.pipeline = pipeline

    def set_new_latent_and_generate_new_image(self, seed=None, prompt=None, negative_prompt="", num_inference_steps=25, guidance_scale=5.0):
        if seed is None: 
            assert False, "Must have a pre-defined random seed"

        if prompt is None:
            assert False, "Must have a user-specified text prompt"

        setup_seed(seed)
        self.latents = torch.randn((1, 4*64*64), device=self.device).to(dtype=self.weight_dtype)
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        prompts = [prompt]
        negative_prompts = [negative_prompt]

        output = inference_latent(
            self.pipeline,
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.guidance_scale,
            latents=self.latents.detach().clone(),
        )

        image = self.pipeline.decode_latents(output['latent'])

        self.org_image = image

        return image
    
