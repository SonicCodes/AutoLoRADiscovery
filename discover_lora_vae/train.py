import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import sys
sys.path.append('..')

from common.loras import patch_lora
from common.utils import make_weight_vector, augmentations, rand_merge
from common.train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    get_optimizer,
    more_init,
    resume_model
)

from types import SimpleNamespace
from discover_lora_vae.models import LoraVAE
from torch.utils.data import Dataset
import random

import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/ubuntu/AutoLoRADiscovery')


from common.render import render_from_lora, render_from_lora_weights
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_weights(attn_weights, max_batches=None, max_heads=None):
    """
    Plot attention weights from a multi-head attention mechanism.
    
    Parameters:
    - attn_weights (torch.Tensor): Attention weights tensor of shape [B, H, S, S]
    - max_batches (int, optional): Maximum number of batches to plot. If None, plot all batches.
    - max_heads (int, optional): Maximum number of heads to plot. If None, plot all heads.
    
    Returns:
    - None (displays the plot)
    """
    
    # Ensure input is a tensor
    if not isinstance(attn_weights, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    # Check input shape
    if len(attn_weights.shape) != 4:
        raise ValueError("Input tensor must have 4 dimensions [B, H, S, S]")
    
    B, H, S, _ = attn_weights.shape
    
    # Limit the number of batches and heads if specified
    B = min(B, max_batches) if max_batches is not None else B
    H = min(H, max_heads) if max_heads is not None else H
    
    # Convert to numpy for plotting
    attn_weights_np = attn_weights[:B, :H].cpu().detach().numpy()
    
    # Create a grid of subplots
    fig, axs = plt.subplots(B, H, figsize=(4*H, 4*B))
    
    # Ensure axs is 2D
    if B == 1 and H == 1:
        axs = np.array([[axs]])
    elif B == 1:
        axs = axs.reshape(1, -1)
    elif H == 1:
        axs = axs.reshape(-1, 1)
    
    # Plot each attention head for each batch
    for b in range(B):
        for h in range(H):
            sns.heatmap(attn_weights_np[b, h], ax=axs[b, h], cmap='viridis', cbar=True)
            axs[b, h].set_title(f'Batch {b+1}, Head {h+1}')
            axs[b, h].set_xlabel('Key')
            axs[b, h].set_ylabel('Query')
            
            # Remove tick labels to save space
            axs[b, h].set_xticks([])
            axs[b, h].set_yticks([])
    
    # Add a colorbar to the right of the subplots
    # fig.colorbar(axs[0, 0].collections[0], ax=axs, location='right', shrink=0.8)
    
    # plt.tight_layout()
    #plt.show()



class LoraDataset(Dataset):

    def __init__(
        self,
        lora_bundle_path,
        num_dataloader_repeats=20, # this could blow up memory be careful!
    ):
        self.lora_bundle  = np.memmap("/mnt/rd/all_weights_recon.npy", dtype='float32', mode='r', shape=(64974, 99648))
        # self.lora_bundle = [make_weight_vector(state_dict) for state_dict in self.lora_bundle]
        # self.weight_dict = self.lora_bundle[0][1]
        # self.lora_bundle = [x[0] for x in self.lora_bundle] * num_dataloader_repeats
        # random.shuffle(self.lora_bundle)

    def __len__(self):
        return len(self.lora_bundle)

    def __getitem__(self, index):
        return torch.Tensor(self.lora_bundle[index])





def collate_fn(examples):
    return torch.stack(examples)


def get_dataset(args):
    train_dataset = LoraDataset(args.data_dir)
    # train_dataset = DummyDataset(1000)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    return train_dataset, train_dataloader, num_update_steps_per_epoch


default_arguments = dict(
    data_dir="/home/ubuntu/AutoLoRADiscovery/lora_bundle.pt",
    output_dir="/mnt/rd/model-out-3",
    seed=None,
    train_batch_size=128,
    max_train_steps=70_000,
    num_dataloader_repeats=100,
    checkpointing_steps=1000,
    # resume_from_checkpoint="/mnt/rd/model-out-2/checkpoint-45000",
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=3.0e-4,
    lr_scheduler="linear",
    lr_warmup_steps=500,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=3,
    use_8bit_adam=True,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_weight_decay=5e-4,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,

    data_dim = 99_648,

    kld_weight = 0.003,

    lora_std = 0.0152,
    use_wandb=True
)


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    lora_vae = LoraVAE(
        input_dim=args.data_dim,
        latent_dim=4096,
    )

    params_to_optimize = list(lora_vae.parameters())
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)
    optimizer, lr_scheduler = get_optimizer(args, params_to_optimize, accelerator)
    # weight_dict = train_dataset.weight_dict



    # Prepare everything with our `accelerator`.
    lora_vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = resume_model(lora_vae, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                    train_dataset, logger, num_update_steps_per_epoch, 
                                                    global_step, wandb_name="lora_vae")

    def train_step(batch):

        pred, z, kld, atten_weights = lora_vae(batch)
        mse_loss = F.mse_loss(pred.float(), batch.float(), reduction="mean")

        split_pred = lora_vae.split(pred)
        split_batch = lora_vae.split(batch)

        losses = []
        for spred, sbatch in zip(split_pred, split_batch):
            losses.append(F.mse_loss(spred.float(), sbatch.float(), reduction="mean"))
        
        losses = torch.stack(losses)
        local_losses = torch.sum(losses) #* 2

        # kld = torch.mean(-0.5 * torch.mean(1 + logvar - mean.float().pow(2) - logvar.float().exp(), dim = 1))
        # kld = mse_loss
        loss = local_losses  + (kld * args.kld_weight)#+ (kld * 4.0) # kld_weight + sp_loss

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        return mse_loss, kld, grad_norm, pred, z, losses, local_losses, atten_weights

    # train_step = torch.compile(train_step)

    grad_norm = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(lora_vae):
                batch = batch.to(accelerator.device)
                batch = lora_vae.apply_std_on_weights(batch)
                mse_loss, kld, grad_norm, pred, pred_z, losses, local_losses, atten_weights = train_step(batch)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        torch.save(unwrap_model(accelerator, lora_vae).state_dict(), save_path)

        

            logs = {
                "mse_loss": mse_loss.detach().item(), 
                "mse_loss_local": local_losses.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "grad_norm": grad_norm.item(), 
            "kld": kld.detach().item(),
            "min_batch": batch.min().item(),
            "max_batch": batch.max().item(),
            "pred_min": pred.min().item(),
            "pred_max": pred.max().item(),

            }#, "sp_loss": sp_loss.detach().item(), "kld": kld.detach().item()}
            
            progress_bar.set_postfix(**logs)
            if (global_step) % 200 == 0:
                plt.plot(pred_z[0].cpu().detach().numpy())
                plt.plot(pred_z[1].cpu().detach().numpy())
                logs["pred_z"] = wandb.Image(plt)
                
                plt.clf()

                plt.plot(losses.cpu().detach().numpy())
                logs["losses"] = wandb.Image(plt)
                
                plt.clf()

                (enc_atten_weights,) = atten_weights
                # print("enc_atten_weights", enc_atten_weights.shape)
                # print("dec_atten_weights", dec_atten_weights.shape)
                plot_attention_weights(enc_atten_weights, max_batches=2, max_heads=8)
                logs["enc_atten_weights"] = wandb.Image(plt)

                plt.clf()

                # plot_attention_weights(dec_atten_weights, max_batches=2, max_heads=8)
                # logs["dec_atten_weights"] = wandb.Image(plt)

                # plt.clf()

            if (global_step) % 1000 == 0:
            
                # lora_diffusion.eval()
                pred_1 = render_from_lora_weights(lora_vae.deapply_std_on_weights(pred[0].unsqueeze(0)).squeeze(0), "pred_1")
                batch_1 = render_from_lora_weights(lora_vae.deapply_std_on_weights(batch[0].unsqueeze(0)).squeeze(0), "batch_1")

                pred_2 = render_from_lora_weights(lora_vae.deapply_std_on_weights(pred[1].unsqueeze(0)).squeeze(0), "pred_2")
                batch_2 = render_from_lora_weights(lora_vae.deapply_std_on_weights(batch[1].unsqueeze(0)).squeeze(0), "batch_2")



                logs["test_1"] = [batch_1, pred_1]
                logs["test_2"] = [batch_2, pred_2]
                # plot pred_z distribution to wandb, pred_z[0] and pred_z[1]
                

            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                pass

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_vae.pth")
        torch.save(unwrap_model(accelerator, lora_vae).state_dict(), save_path)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)