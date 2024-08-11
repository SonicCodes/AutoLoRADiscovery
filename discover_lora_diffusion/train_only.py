import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import random
import sys
sys.path.append('/home/ubuntu/AutoLoRADiscovery')


from common.render import render_from_lora, render_from_lora_weights
from common.train_utils import (
    init_train_basics,
    unwrap_model,
    get_optimizer,
    more_init,
    resume_model
)

from types import SimpleNamespace
from discover_lora_diffusion.models import LoraDiffusion
from torch.utils.data import Dataset
import random
import diffusers
import numpy as np

run_prop= "clpface_newdataset22_small_50step_denoise_reluflat"


class LoraDataset(Dataset):
    def __init__(
        self,
    ):
        self.ldd = list(torch.load("/mnt/rd/celeba_vae_map_clip_arc.pt", map_location="cpu").items())
        self.lora_bundle  = np.memmap("/mnt/rd/all_weights_recon.npy", dtype='float32', mode='r', shape=(64974, 99648))

        random.shuffle(self.ldd)

    def __len__(self):
        return len(self.ldd)

    def __getitem__(self, index):
        (file, (z, face_embedding, properties, idx)) = self.ldd[index]
        # split the mean_logvar
        # mean, logvar = mean_logvar.chunk(2, dim=-1)
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # z = mean + eps * std
        return torch.Tensor(self.lora_bundle[idx]), face_embedding[0].flatten().float(), face_embedding[1].flatten().float(), file, torch.Tensor((properties[:-1]))


device = "cuda"
def collate_fn(examples):
    lora_bundles, clip_embeds, face_embeddings, files, properties = zip(*examples)
    return torch.stack(lora_bundles), torch.stack(clip_embeds), torch.stack(face_embeddings), files, torch.stack(properties)

def get_dataset(args):
    train_dataset = LoraDataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    return train_dataset, train_dataloader, num_update_steps_per_epoch




default_arguments = dict(
    data_dir="/home/ubuntu/AutoLoRADiscovery/loras2.pt",
    output_dir="/mnt/rd/diff_lora_mdl",
    seed=None,
    train_batch_size=128,
    max_train_steps=100_000,
    # validation_steps=250,
    num_dataloader_repeats=300,
    checkpointing_steps=2000,
    resume_from_checkpoint=None,#"/mnt/rd/diff_lora_mdl/checkpoint-clpface_newdataset22_small_50step_denoise_reluflat-98000",#"/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/diffusion_lora/checkpoint-76000",
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=1.0e-4,
    lr_scheduler="linear",
    lr_warmup_steps=500,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_weight_decay=1e-3,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,
    snr_gamma=None,
    lora_std = 0.0152,
    use_wandb=True
    
    
)

from discover_lora_vae.models import LoraVAE
# lora_vae = None
lora_vae = LoraVAE(
    input_dim=99_648,
    latent_dim=4096,
).cuda()

lora_vae.load_state_dict(torch.load("/mnt/rd/model-out-2/checkpoint-40000", map_location="cuda"))


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    lora_diffusion = LoraDiffusion(
        data_dim=4096,
        model_dim=2048,
        ff_mult=2,
        chunks=1,
        act=torch.nn.SiLU,
        num_blocks=3,
        layers_per_block=3,
    )

    # scheduler = diffusers.UnCLIPScheduler.from_config("kandinsky-community/kandinsky-2-2-prior", subfolder="scheduler")
    config = {
        "_class_name": "UnCLIPScheduler",
        "_diffusers_version": "0.17.0.dev0",
        "clip_sample": True,
        "clip_sample_range": 10.0,  # Adjusted to match your data range
        "num_train_timesteps": 100,  # Reduced from 1000
        "prediction_type": "sample",
        "variance_type": "fixed_small_log"
    }

    scheduler = diffusers.UnCLIPScheduler.from_config(config)
    # scheduler.set_timesteps(50)
    # scheduler = diffusers.UnCLIPScheduler.from_config("kandinsky-community/kandinsky-2-2-prior", subfolder="scheduler")

    params_to_optimize = list(lora_diffusion.parameters())
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)
    optimizer, lr_scheduler = get_optimizer(args, params_to_optimize, accelerator)


    # Prepare everything with our `accelerator`.
    lora_diffusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_diffusion, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = resume_model(lora_diffusion, args.resume_from_checkpoint, accelerator)


    # generate random number from 0 to 100
    number_ = random.randint(0, 10)
    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, 
                                                        global_step, wandb_name="diffusion_lora", wandb_runname=run_prop+f"_{number_}")

    # zip and store the following files: into a folder called "./archives/run_prop+f"-{number_}""
    
    storable_files = ["/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/models.py", "/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/train.py", "/home/ubuntu/AutoLoRADiscovery/common/utils.py"]
    if not os.path.exists(f"/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/archives/{run_prop}_{number_}"):
        os.makedirs(f"/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/archives/{run_prop}_{number_}")
    for file in storable_files:
        
        os.system(f"cp {file} /home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/archives/{run_prop}_{number_}/")

    std_w2w = torch.load("/mnt/rd/std_w2w.pt", map_location="cuda") * 4.0

    print("number of steps", scheduler.config.num_train_timesteps)
    # ixx = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        lora_diffusion.train()
        for step, (batch, clip_embs, face_embeddings, files, properties) in enumerate(train_dataloader):
            with accelerator.accumulate(lora_diffusion):
                batch = batch.to(accelerator.device) #* (1 / args.lora_std)
                properties = properties.to(accelerator.device)
                noise = torch.randn((batch.shape[0], 4096)).to(accelerator.device)
                with torch.no_grad():
                    batch = lora_vae.encode(lora_vae.apply_std_on_weights(batch), actual_noise=noise)[0]
                    batch = lora_vae.standardized_z(batch)
                # print("properties.shape=", (properties).shape)
                # print("properties.min,max,mean,std", properties.min().item(), properties.max().item(), properties.mean().item(), properties.std().item())
                # normalize clip embeddings
                clip_embs = clip_embs.to(accelerator.device)
                clip_embs = clip_embs / clip_embs.norm(dim=-1, keepdim=True)
                # clip_embs = (clip_embs - clip_embs.mean(dim=-1, keepdim=True)) / clip_embs.std(dim=-1, keepdim=True)
                # clip_embs = (clip_embs * 0.200) + (-0.0002)
                # clip_embs = clip_embs * 2.3
                # merge batch and clip embeddings

                # # print("min(batch)", batch.min().item(), " max(batch)", batch.max().item(), ' mean(batch)', batch.mean().item(), " std(batch)", batch.std().item())
                # # print("min(clip_embs)", clip_embs.min().item(), " max(clip_embs)", clip_embs.max().item(), ' mean(clip_embs)', clip_embs.mean().item(), " std(clip_embs)", clip_embs.std().item())

                # print("min(batch)", batch.min().item(), " max(batch)", batch.max().item(), ' mean(batch)', batch.mean().item(), " std(batch)", batch.std().item())
                # # print("min(clip_embs)", clip_embs.min().item(), " max(clip_embs)", clip_embs.max().item(), ' mean(clip_embs)', clip_embs.mean().item(), " std(clip_embs)", clip_embs.std().item())
                # raise ValueError("stop here")
                # batch = torch.cat([batch, clip_embs], dim=-1)
                face_embedding = face_embeddings.to(accelerator.device).detach()

                
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=batch.device
                ).long()

                noisy_model_input = scheduler.add_noise(batch, noise, timesteps)
                # prev_noisy_model_input = scheduler.add_noise(batch, noise, torch.relu(timesteps-1))

                # print ("min(noisy_model_input)", noisy_model_input.min().item(), " max(noisy_model_input)", noisy_model_input.max().item(), ' mean(noisy_model_input)', noisy_model_input.mean().item(), " std(noisy_model_input)", noisy_model_input.std().item())
                # print ("min(batch)", batch.min().item(), " max(batch)", batch.max().item(), ' mean(batch)', batch.mean().item(), " std(batch)", batch.std().item())
                # raise ValueError("stop here")
                # 10% of the time we'll drop the conditioning
                # dropped_cond = False
                _face_embedding= face_embedding 

                # normalize the face_embedding
                _face_embedding_norm = _face_embedding / _face_embedding.norm(dim=-1, keepdim=True)
                # if random.random() <= 0.1:
                #     properties = torch.zeros_like(properties)
               

                # assert if there are no 
                # dropped_cond = True
                # print("face_embedding.shape", _face_embedding.shape)
                
                pred, pred_cond = lora_diffusion(noisy_model_input, timesteps, properties)
                
                # weight_relevance = (std_w2w.log()/10.0).detach()
               

                mse_loss = F.mse_loss(pred, batch, reduction="mean")
                # lora_mse = F.mse_loss(pred[:, :10_000], batch[:, :10_000], reduction="mean")
                # cond_mse = F.mse_loss(pred[:, 10_000:], batch[:, 10_000:], reduction="mean")

                # print ("min(pred)", pred.min().item(), " max(pred)", pred.max().item(), ' mean(pred)', pred.mean().item(), " std(pred)", pred.std().item())
                # # print ("min(batch)", batch.min().item(), " max(batch)", batch.max().item(), ' mean(batch)', batch.mean().item(), " std(batch)", batch.std().item())
                # # print ("cond_mse", cond_mse.item(), "lora_mse", lora_mse.item())
                # # print("min(pred cond)", pred[10_000:].min().item(), " max(pred cond)", pred[10_000:].max().item(), ' mean(pred cond)', pred[10_000:].mean().item(), " std(pred cond)", pred[10_000:].std().item())

                # # raise ValueError("stop here")

                # mse_loss = lora_mse #+ cond_mse
                # pred_cond = pred_cond 
                total_clip_loss = None#F.mse_loss(pred_cond, properties, reduction="mean") #* 0.0
                # pred_cond = True
                # if  is not None:
                #     pass
                #     # pred_cond = pred_cond / pred_cond.norm(dim=-1, keepdim=True)
                #     # mse_embed_loss = F.mse_loss(pred_cond, _face_embedding, reduction="mean")
                #     # total_clip_loss = (mse_embed_loss) #+ cosine_embedding_loss
                # else:
                #     total_clip_loss = None
               

                loss= mse_loss + (total_clip_loss if ((pred_cond is not None)) else 0.0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{run_prop}-{global_step}")
                        torch.save(unwrap_model(accelerator, lora_diffusion).state_dict(), save_path)

            

            logs = {
                "mse_loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], 
                # "cond_recon_loss": total_clip_loss.detach().item() if total_clip_loss is not None else 0.0,
                "grad_norm": grad_norm.item() if accelerator.sync_gradients else 0.0,
            }

            if total_clip_loss is not None:
                logs["cond_recon_loss"] = total_clip_loss.detach().item()
            if (global_step - 1) % 500 == 0:
                lora_diffusion.eval()
                rend_r, _, (ood_main_comps, ood_cross_comps) = render_from_lora(lora_diffusion, scheduler, lora_vae=lora_vae)
                rend_b, _, _ = render_from_lora(lora_diffusion, scheduler, batch[0].unsqueeze(0), files[0], lora_vae=lora_vae)
                rend_d, _, (train_main_comps, train_cross_comps) = render_from_lora(lora_diffusion, scheduler, fake_face_embedding=face_embeddings[0], custom_title=files[0], lora_vae=lora_vae)
                logs["ood_face"] = rend_r
                logs["from_face_train"] = rend_d
                logs["latent_train"] = rend_b
                logs["ood_main_comps"] = ood_main_comps
                logs["ood_cross_comps"] = ood_cross_comps
                logs["train_main_comps"] = train_main_comps
                logs["train_cross_comps"] = train_cross_comps
                # logs["valid_cond_recon_loss"] = cosine_sim_val.detach().item()
                lora_diffusion.train()
            progress_bar.set_postfix(**logs)
            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                pass

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_diffusion.pth")
        torch.save(unwrap_model(accelerator, lora_diffusion).state_dict(), save_path)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)