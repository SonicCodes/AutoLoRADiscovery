import torch
import safetensors
from .lora_resize import svd, get_least_squares_solution, change_lora_rank
import requests
import pandas as pd
import re
import glob
from pathlib import Path
import torch.nn.functional as F
import torch
import random

def open_weights(path):
    try: # if ".safetensors" in path:
        state_dict = {}
        with safetensors.safe_open(path, "pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    except:
        state_dict = torch.load(path, map_location="cpu")

    return state_dict


def get_num_params(state_dict):
    return sum([torch.numel(v) for v in state_dict.values()])


def convert_to_multi(state_dict, idx=0):
    new_state_dict = {}
    for k,v in state_dict.items():

        # single lora method to multi
        k = k.replace(".weight", f".{idx}")
        if k.endswith("lora_weight"):
            k = k.replace("lora_weight", f"lora_weight.{idx}")
        if k.endswith("lora_bias"):
            k = k.replace("lora_bias", f"lora_bias.{idx}")

        # changing index 
        k = re.sub(r"lora_down\.(\d+)", f"lora_down.{idx}", k)
        k = re.sub(r"lora_up\.(\d+)", f"lora_up.{idx}", k)
        k = re.sub(r"lora_weight\.(\d+)", f"lora_weight.{idx}", k)
        k = re.sub(r"lora_bias\.(\d+)", f"lora_bias.{idx}", k)

        new_state_dict[k] = v
    return new_state_dict



def aggregate_loras(path, rank):
    files = glob.glob(f"{path}/**/lora_layers.pth", recursive=True)
    all_state_dicts = []
    for f in files:
        # basically runs svd on it, this ensures down/up weights are balanced
        state_dict = torch.load(f)
        state_dict = change_lora_rank(state_dict, rank=rank)
        all_state_dicts.append(state_dict)
    return all_state_dicts


# def aggregate_loras(path, rank):
#     files = glob.glob(f"{path}/**/lora_layers.pth", recursive=True)
#     for f in tqdm(files):
#         # basically runs svd on it, this ensures down/up weights are balanced
#         state_dict = torch.load(f, map_location="cuda")
#         new_state_dict = {}
#         new_state_dict["state_dict"] = {}
#         new_state_dict["state_dict"].update(state_dict['unet'])
#         new_state_dict["state_dict"].update(state_dict['text_encoder'])
#         new_state_dict['num_params'] = state_dict['num_params']
#         new_state_dict['lora_layers'] = state_dict['lora_layers']
#         new_state_dict['lora_layers_te'] = state_dict['lora_layers_te']
#         new_state_dict = change_lora_rank(new_state_dict, rank=rank)
#         torch.save(new_state_dict, f.replace("lora_layers","lora_layers_new"))

    
def make_weight_vector(state_dict):
    # ensure same ordering
    keys = sorted(list(state_dict.keys()))
    weight_dict = {}
    weights = []
    idx = 0
    for k in keys:
        weight = state_dict[k]
        if isinstance(weight, torch.Tensor):
            shape = weight.shape
            flattened = weight.flatten()
            interval = (idx, idx + len(flattened), shape)
            idx = idx + len(flattened)
            weight_dict[k] = interval
            weights.append(flattened)
    
    return torch.cat(weights), weight_dict


# def slerp(a, b, t):
#     omega = torch.acos((a / a.norm()).dot(b / b.norm()))
#     so = torch.sin(omega)
#     return torch.sin((1.0 - t) * omega) / so * a + torch.sin(t * omega) / so * b


def batch_slerp(a, b, t):
    if len(t.shape) == 1:
        t = t[..., None]
    dot = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)[...,None].clamp(-0.999999, 0.999999)
    # dot = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)[...,None].clamp(-0.9999, 0.9999)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    part1 = torch.sin((1.0 - t) * omega) / sin_omega * a
    part2 = torch.sin(t * omega) / sin_omega * b
    return part1 + part2


def recover_lora(lora_vector, weight_dict):
    state_dict = {}
    prev_idx = 0
    for k, stuff in weight_dict.items():
        start = stuff[0]
        end = stuff[1]
        shape = stuff[2]
        assert start >= prev_idx
        state_dict[k] = lora_vector[0, start:end].view(shape)
        prev_idx = end
    return state_dict


def lerp(a, b, t):
    return a * (1 - t) + b * t


@torch.no_grad()
def rand_merge_layerwise(lora_a, lora_b, weight_dict, slerp=True):
    b, n = lora_a.shape
    interp_fn = batch_slerp if slerp else lerp
    out = torch.zeros(b, n).cuda()
    for k in weight_dict.keys():
        start = weight_dict[k][0]
        end = weight_dict[k][1]
        coef = torch.rand(b, 1).expand(b, end-start).cuda()
        # out[:, start:end] = batch_slerp(lora_a[:, start:end], lora_b[:, start:end], coef)
        out[:, start:end] = interp_fn(lora_a[:, start:end], lora_b[:, start:end], coef)

    return out


@torch.no_grad()
def getcoeff(lora_a, lora_b, slerp=True):
    coef = torch.rand(lora_a.shape[0], 1).cuda().to(lora_a.dtype)
    return coef
@torch.no_grad()
def rand_merge(lora_a, lora_b, slerp=True, coef=None):
    interp_fn = batch_slerp if slerp else lerp
    if (not torch.is_tensor(coef)):
        coef = torch.rand(lora_a.shape[0], 1).cuda().to(lora_a.dtype)
    # out = batch_slerp(lora_a, lora_b, coef)
    out = interp_fn(lora_a, lora_b, coef)
    return out



def augmentations2(batch, face_embeddings, weight_dict,
                  first=0.85,
                  second=0.35,
                  second_w_orig=0.5,
                  third=0.1,
                  third_w_orig=0.5,
                  layerwise=0.01,
                  slerp=True,
                  null_p=0.001):
    orig_batch = batch.clone()
    orig_face_embeddings = face_embeddings.clone()

    # Create masks to identify samples with valid face embeddings
    valid_mask = (face_embeddings != 0).any(dim=-1).cpu()

    # Apply augmentations to samples with valid face embeddings
    mask = torch.rand(batch.shape[0]) < first
    valid_mask_indices = mask.nonzero(as_tuple=True)[0][valid_mask[mask]]
    if valid_mask_indices.numel() > 0:
        co_eff = getcoeff(orig_batch[valid_mask_indices], batch.flip(0)[valid_mask_indices])
        batch[valid_mask_indices] = rand_merge(orig_batch[valid_mask_indices], batch.flip(0)[valid_mask_indices], coef=co_eff)
        face_embeddings[valid_mask_indices] = rand_merge(orig_face_embeddings[valid_mask_indices], 
                                                         face_embeddings.flip(0)[valid_mask_indices], 
                                                         True,
                                                         coef=co_eff)

    mask = torch.rand(batch.shape[0]) < second
    valid_mask_indices = mask.nonzero(as_tuple=True)[0][valid_mask[mask]]
    if valid_mask_indices.numel() > 0:
        perm = torch.randperm(valid_mask_indices.numel())
        if random.random() < second_w_orig:
            co_eff = getcoeff(batch[valid_mask_indices], orig_batch[valid_mask_indices[perm]])
            batch[valid_mask_indices] = rand_merge(batch[valid_mask_indices], orig_batch[valid_mask_indices[perm]], slerp, coef=co_eff)
            face_embeddings[valid_mask_indices] = rand_merge(face_embeddings[valid_mask_indices], orig_face_embeddings[valid_mask_indices[perm]], slerp, coef=co_eff)
        else:
            co_eff = getcoeff(batch[valid_mask_indices], batch[valid_mask_indices[perm]])
            batch[valid_mask_indices] = rand_merge(batch[valid_mask_indices], batch[valid_mask_indices[perm]], slerp, coef=co_eff)
            face_embeddings[valid_mask_indices] = rand_merge(face_embeddings[valid_mask_indices], face_embeddings[valid_mask_indices[perm]], slerp, coef=co_eff)

    # ... (similar modifications for third and layerwise augmentations) ...
    mask = torch.rand(batch.shape[0]) < layerwise
    if sum(mask) > 0:
        perm = torch.randperm(batch.shape[0])
        batch[mask] = rand_merge_layerwise(batch[mask], orig_batch[perm][mask], weight_dict, slerp)
    # Handle samples with zero face embeddings
    zero_mask = ~valid_mask
    mask = torch.rand(batch.shape[0]) < null_p
    zero_mask_indices = mask.nonzero(as_tuple=True)[0][zero_mask[mask]]
    if zero_mask_indices.numel() > 0:
        batch[zero_mask_indices] = orig_batch[zero_mask_indices]
        face_embeddings[zero_mask_indices] = torch.zeros_like(face_embeddings[zero_mask_indices])

    return batch, face_embeddings

# def augmentations2(batch, 
#                    face_embeddings1, 
#                      face_embeddings2, 
#                    weight_dict,
#                   first=0.85,
#                   second=0.35,
#                   second_w_orig=0.5,
#                   third=0.1,
#                   third_w_orig=0.5,
#                   layerwise=0.01,
#                   slerp=True,
#                   null_p=0.001):
#     orig_batch = batch.clone()
#     orig_face_embeddings1 = face_embeddings1.clone()
#     orig_face_embeddings2 = face_embeddings2.clone()

#     # Create masks to identify samples with valid face embeddings
#     # valid_mask = (face_embeddings != 0).any(dim=-1).cpu()

#     # Apply augmentations to samples with valid face embeddings
#     mask = torch.rand(batch.shape[0]) < first
#     valid_mask_indices = mask.nonzero(as_tuple=True)[0]#[valid_mask[mask]]
#     if valid_mask_indices.numel() > 0:
#         co_eff = getcoeff(orig_batch[valid_mask_indices], batch.flip(0)[valid_mask_indices])
#         batch[valid_mask_indices] = rand_merge(orig_batch[valid_mask_indices], batch.flip(0)[valid_mask_indices], coef=co_eff)
#         face_embeddings1[valid_mask_indices] = rand_merge(orig_face_embeddings1[valid_mask_indices], 
#                                                          face_embeddings1.flip(0)[valid_mask_indices], 
#                                                          True,
#                                                          coef=co_eff)
#         face_embeddings2[valid_mask_indices] = rand_merge(orig_face_embeddings2[valid_mask_indices], 
#                                                          face_embeddings2.flip(0)[valid_mask_indices], 
#                                                          True,
#                                                          coef=co_eff)

#     mask = torch.rand(batch.shape[0]) < second
#     valid_mask_indices = mask.nonzero(as_tuple=True)[0]#[valid_mask[mask]]
#     if valid_mask_indices.numel() > 0:
#         perm = torch.randperm(valid_mask_indices.numel())
#         if random.random() < second_w_orig:
#             co_eff = getcoeff(batch[valid_mask_indices], orig_batch[valid_mask_indices[perm]])
#             batch[valid_mask_indices] = rand_merge(batch[valid_mask_indices], orig_batch[valid_mask_indices[perm]], slerp, coef=co_eff)
#             face_embeddings1[valid_mask_indices] = rand_merge(face_embeddings1[valid_mask_indices], orig_face_embeddings1[valid_mask_indices[perm]], slerp, coef=co_eff)
#             face_embeddings2[valid_mask_indices] = rand_merge(face_embeddings2[valid_mask_indices], orig_face_embeddings2[valid_mask_indices[perm]], slerp, coef=co_eff)
#         else:
#             co_eff = getcoeff(batch[valid_mask_indices], batch[valid_mask_indices[perm]])
#             batch[valid_mask_indices] = rand_merge(batch[valid_mask_indices], batch[valid_mask_indices[perm]], slerp, coef=co_eff)
#             face_embeddings1[valid_mask_indices] = rand_merge(face_embeddings1[valid_mask_indices], face_embeddings1[valid_mask_indices[perm]], slerp, coef=co_eff)
#             face_embeddings2[valid_mask_indices] = rand_merge(face_embeddings2[valid_mask_indices], face_embeddings2[valid_mask_indices[perm]], slerp, coef=co_eff)


#     return batch, face_embeddings1, face_embeddings2

def augmentations(batch, weight_dict, 
                    first=0.85, 
                    second=0.35,
                    second_w_orig=0.5,
                    third=0.1, 
                    third_w_orig=0.5,
                    layerwise=0.01,
                    slerp=True,
                    null_p=0.001,
                    ):
    orig_batch = batch.clone()
    mask = torch.rand(batch.shape[0]) < first
    if sum(mask) > 0:
        batch[mask] = rand_merge(orig_batch[mask], batch.flip(0)[mask])


    mask = torch.rand(batch.shape[0]) < second
    if sum(mask) > 0:
        perm = torch.randperm(batch.shape[0])
        if random.random() < second_w_orig:
            batch[mask] = rand_merge(batch[mask], orig_batch[perm][mask], slerp)
        else:
            batch[mask] = rand_merge(batch[mask], batch[perm][mask], slerp)
    
    mask = torch.rand(batch.shape[0]) < third
    if sum(mask) > 0:
        perm = torch.randperm(batch.shape[0])
        if random.random() < third_w_orig:
            batch[mask] = rand_merge(batch[mask], orig_batch[perm][mask], slerp)
        else:
            batch[mask] = rand_merge(batch[mask], batch[perm][mask], slerp)

    mask = torch.rand(batch.shape[0]) < layerwise
    if sum(mask) > 0:
        perm = torch.randperm(batch.shape[0])
        batch[mask] = rand_merge_layerwise(batch[mask], orig_batch[perm][mask], weight_dict, slerp)

    mask = torch.rand(batch.shape[0]) < null_p
    if sum(mask) > 0:
        batch[mask] = orig_batch[mask]

    return batch





