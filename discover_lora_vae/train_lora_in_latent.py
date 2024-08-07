#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import sys
sys.path.append('..')

from common.utils import make_weight_vector, augmentations, recover_lora
from common.train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    more_init,
    resume_model,
    default_arguments,
    load_models,
    get_optimizer,
    get_dataset,
)

from types import SimpleNamespace
from discover_lora_vae.models import LoraVAE
from torch.utils.data import Dataset
import random

from discover_lora_vae.dynamic_lora import give_weights, patch_lora

def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    args.instance_prompt= f"A photo of {args.subject}"
    args.validation_prompt = [f"majestic fantasy painting of {args.subject}", f"a comic book drawing of {args.subject}", f"HD cinematic photo of {args.subject}", f"oil painting of {args.subject} by van gogh"]
    accelerator, weight_dtype = init_train_basics(args, logger)

    lora_bundle_path ='/home/ubuntu/AutoLoRADiscovery/lora_bundle.pt'

    lora_vae = LoraVAE(data_dim=1_365_504,
                        model_dim=256,
                        ff_mult=3.0,
                        chunks=1,
                        encoder_layers=20,
                        decoder_layers=20
                        ).requires_grad_(False)

    lora_bundle = torch.load(lora_bundle_path)
    _, weight_dict = make_weight_vector(lora_bundle[0])

    lora_vae_state_dict_path = "/home/ubuntu/AutoLoRADiscovery/discover_lora_vae/model-output/checkpoint-45000"
    lora_vae.load_state_dict(torch.load(lora_vae_state_dict_path, map_location="cpu"))

    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(args, accelerator, weight_dtype)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # now we will add new LoRA weights to the attention layers
    patch_lora(unet, rank=args.lora_rank, included_terms=args.lora_layers)

    # The text encoder comes from transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        patch_lora(text_encoder, rank=args.lora_rank, included_terms=args.lora_layers_te)

    # Optimizer creation
    lora_latent = torch.randn(1, min(lora_vae.decoder.in_proj.weight.shape)).to(accelerator.device).requires_grad_(True)

    optimizer, lr_scheduler = get_optimizer(args, [lora_latent], accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args, tokenizer)

    # Prepare everything with our `accelerator`.
    unet, text_encoder, lora_vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, lora_vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # if args.resume_from_checkpoint:
    #     global_step = resume_model(unet, args.resume_from_checkpoint, accelerator)
    #     global_step = resume_model(text_encoder, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="lora_in_latent")

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_latent):
                with torch.no_grad():
                    model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(model_input)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=model_input.device
                    ).long()
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                lora = lora_vae.decoder(lora_latent)

                ########
                # not currently working with autograd
                lora_dict = recover_lora(lora, weight_dict)
                give_weights(unet, lora_dict)
                give_weights(text_encoder, lora_dict)
                ########

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device),return_dict=False,)[0]

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(lora_latent, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(unet, text_encoder,accelerator,save_path, args, logger)
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                # print(global_step, global_step % args.validation_steps)
                if args.validation_prompt is not None and global_step % args.validation_steps == 0 and global_step > 0:
                    images = log_validation(
                        unet,
                        text_encoder,
                        weight_dtype,
                        args,   
                        accelerator,
                        pipeline_args={"prompt": args.validation_prompt, "height": args.resolution, "width": args.resolution},
                        epoch=epoch,
                        logger=logger,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_layers.pth")
        save_model(unet, text_encoder, accelerator, save_path, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)