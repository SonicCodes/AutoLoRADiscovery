import torch
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align
import cv2
import numpy as np
import sys
import sys
import random
import os 
sys.path.append(os.path.abspath(os.path.join("", "..")))
sys.path.append("/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/weights2weights/")
import torch
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from discover_lora_diffusion.weights2weights.lora_w2w import LoRAw2w
from discover_lora_diffusion.weights2weights.utils import unflatten
from diffusers import DiffusionPipeline 
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights
# 

# import clip
import wandb
# import clip
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

# from transformers import AutoImageProcessor, AutoModel
# preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
# dino_model = AutoModel.from_pretrained('facebook/dinov2-large')
# dino_model = dino_model.cuda()

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

def rtn_face_get(self, img, face):
    aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    #print(cv2.imwrite("aimg.png", aimg))
    face.embedding = self.get_feat(aimg).flatten()
    face.crop_face = aimg
    return face.embedding

ArcFaceONNX.get = rtn_face_get
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def crop_face(img, face_ratio=1.0):
    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = app.get(cv2_img)
    try:
        bbox = faces[0]['bbox']

        # get largest possible square around center point
        w,h = img.width, img.height
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        if box_w < box_h:
            diff = box_h - box_w
            bbox[0] -= diff // 2
            bbox[2] += diff // 2
        else:
            diff = box_w - box_h
            bbox[1] -= diff // 2
            bbox[3] += diff // 2
        dist_to_left = bbox[0]
        dist_to_right = w - bbox[2]
        dist_to_top = bbox[1]
        dist_to_bottom = h - bbox[3]
        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        min_dist = int(min_dist * face_ratio) - 1
        bbox = [bbox[0]-min_dist, bbox[1]-min_dist, bbox[2]+min_dist, bbox[3]+min_dist]
        face_image = img.crop(bbox)

        return face_image, faces[0].embedding
    except:
        return None
    
# Load pre-trained FaceNet model
device = "cuda"


def get_face_embedding(image_path):
    if type(image_path) == str:
        img = Image.open(image_path)
    else:
        img = image_path
    try:
        face, emb_gotten = crop_face(img)#.convert('RGB')
        if face is None:
            return None
        # clip load 
        return torch.Tensor(emb_gotten.flatten()).cuda()
    except:
        return None
    
    with torch.no_grad():
        # face_image = preprocess(img).unsqueeze(0).to(device)
        # full_image = preprocess(face).unsqueeze(0).to(device)

        full_image = preprocess(img).unsqueeze(0).cuda()#.pixel_values#.squeeze(0)

        # total_embs = torch.cat([full_image, face_image], dim=0).to(device)
        emb = clip_model.encode_image(full_image)#.last_hidden_state.mean(dim=1)
    return emb.flatten()


# V = torch.load("/mnt/rd/V.pt", map_location="cpu").to("cuda", torch.bfloat16)
# std = torch.load("/mnt/rd/std.pt", map_location="cuda")
# mean = torch.load("/mnt/rd/mean.pt", map_location="cuda")

weight_dimensions = torch.load("/mnt/rd/weight_dimensions.pt")




#celeba_vae_map_clip_arc
def w2w_to_lora(lora_vae, latent_weights):
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        return lora_vae.deapply_std_on_weights(lora_vae.decode(latent_weights)).cpu()


std_w2w = torch.load("/mnt/rd/std_w2w.pt", map_location="cuda") * 4.0


def generate_denoising_sequence_viz(list_time_latents, location):
    latents = torch.cat(list_time_latents, dim=0)
    latents = latents[:, :36*36*3].view(-1, 36, 36, 3)
    # convert to merged image list
    canvas = Image.new('RGB', (36*latents.shape[0], 36))
    for i, latent in enumerate(latents):
        img = Image.fromarray((latent.cpu().numpy() * 255).astype(np.uint8))
        canvas.paste(img, (36*i, 0))
    # save canvas as jpeg with 20 quality
    canvas.save(location, format="JPEG", quality=20)

idens = torch.load("/mnt/rd/identity_df.pt", map_location="cpu")
def get_label_for_choice(properties):
    cols = idens.columns.tolist()
    labels = []
    for idx in properties:
        labels.append(cols[idx])
    return ", ".join(labels)

feature_combinations = [
    ['Arched_Eyebrows', 'Attractive', 'Big_Lips', 'Narrow_Eyes', 'No_Beard', 'Pointy_Nose', 'Wearing_Lipstick', 'Young'],
    ['5_o_Clock_Shadow', 'Bags_Under_Eyes', 'Big_Nose', 'Bushy_Eyebrows', 'Male', 'Mouth_Slightly_Open', 'Smiling'],
    ['Attractive', 'Blond_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Wearing_Earrings', 'Wearing_Necklace'],
    ['Bald', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Gray_Hair', 'Male', 'Receding_Hairline'],
    ['Bangs', 'Black_Hair', 'Heavy_Makeup', 'Oval_Face', 'Pale_Skin', 'Smiling', 'Wearing_Lipstick'],
    ['Attractive', 'Brown_Hair', 'High_Cheekbones', 'Narrow_Eyes', 'Wavy_Hair', 'Young'],
    ['Arched_Eyebrows', 'Goatee', 'Male', 'Mustache', 'No_Beard', 'Sideburns', 'Wearing_Necktie'],
    ['Bags_Under_Eyes', 'Blurry', 'Pale_Skin', 'Pointy_Nose', 'Straight_Hair', 'Wearing_Hat'],
    ['Attractive', 'Big_Lips', 'Heavy_Makeup', 'High_Cheekbones', 'Rosy_Cheeks', 'Smiling', 'Young'],
    ['5_o_Clock_Shadow', 'Bushy_Eyebrows', 'Chubby', 'Male', 'Narrow_Eyes', 'No_Beard'],
    ['Attractive', 'Blond_Hair', 'Oval_Face', 'Smiling', 'Wearing_Earrings', 'Young'],
    ['Bald', 'Big_Nose', 'Eyeglasses', 'Male', 'No_Beard', 'Pale_Skin'],
    ['Arched_Eyebrows', 'Brown_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Pointy_Nose', 'Wearing_Necklace'],
    ['Bags_Under_Eyes', 'Gray_Hair', 'Male', 'Receding_Hairline', 'Smiling', 'Wearing_Necktie'],
    ['Attractive', 'Bangs', 'Black_Hair', 'Narrow_Eyes', 'No_Beard', 'Wearing_Lipstick'],
    ['Chubby', 'Double_Chin', 'Mouth_Slightly_Open', 'Rosy_Cheeks', 'Wavy_Hair'],
    ['5_o_Clock_Shadow', 'Attractive', 'Goatee', 'High_Cheekbones', 'Male', 'Young'],
    ['Blurry', 'Eyeglasses', 'Pale_Skin', 'Straight_Hair', 'Wearing_Hat'],
    ['Big_Lips', 'Bushy_Eyebrows', 'Heavy_Makeup', 'Oval_Face', 'Smiling'],
    ['Attractive', 'Male', 'No_Beard', 'Pointy_Nose', 'Sideburns', 'Young']
]

def get_id_for_choice(properties):
    cols = idens.columns.tolist()
    ids = []
    for idx, col in enumerate(cols):
        if col in properties:
            ids.append(idx)
    return np.array(ids)


def render_from_lora(lora_diffusion, scheduler, custom_latent=None, custom_title=None, fake_face_embedding=None,lora_vae=None):
    global weight_dimensions
    # global pipe
    latents = custom_latent
    face_emb1 = None
    caption = "Default from batch"
    if custom_latent is None:
        if fake_face_embedding is None:
            face_emb1 = torch.Tensor(get_face_embedding("/home/ubuntu/AutoLoRADiscovery/junk/1691527680903.jpeg")).unsqueeze(0)
        else:
            face_emb1 = fake_face_embedding.unsqueeze(0)

        
        face_emb_norm = face_emb1 / face_emb1.norm(dim=-1, keepdim=True)

        properties = torch.ones(40).cuda() - 2
        # make 4 random properties 1
        
        activated_props_ls = random.choice(feature_combinations)
        activated_props = get_id_for_choice(activated_props_ls)
        properties[activated_props] = 1
        properties = properties.unsqueeze(0)
        caption = ", ".join(activated_props_ls)

        
        # print("Rendering from face embedding, shape:", face_emb1.shape)
        latents = torch.randn(1, 4096).cuda().to(torch.float16)  * scheduler.init_noise_sigma 
        
        # Create an unconditioned embedding for CFG
        # uncond_face_emb = torch.zeros_like(face_emb1)

        # prev_latents = [latents]
        # latents, _ = lora_diffusion(face_emb1)
        with torch.no_grad():
            for t in scheduler.timesteps:
                # latents = scheduler.scale_model_input(latents, t)
                noise_pred, pred_cond = lora_diffusion(
                    latents, 
                    t.unsqueeze(0).cuda().half(), 
                    properties
                )
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        #         prev_latents.append(latents)
        # mean, logvar = latents.chunk(2,w  dim=-1) 
        # latents = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        # latents = latents #* 0.0152  #* std_w2w
        # latents = w2w_to_lora(lora_vae, latents)
    else:
        prev_latents = [latents]
    latents = lora_vae.unstandardized_z(latents) #* 2# / 2
    latents = w2w_to_lora(lora_vae,latents)
    # latents =  lora_vae.deapply_std_on_weights(latents)

    if os.path.exists("/mnt/rd/inference_lora"):
        os.system("rm -rf /mnt/rd/inference_lora")

    unflatten(latents.detach().clone(), weight_dimensions, "/mnt/rd/inference_lora")
    
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51", 
                                         torch_dtype=torch.float16,safety_checker = None,
                                         requires_safety_checker = False).to(device)

    
    pipe.unet = PeftModel.from_pretrained(pipe.unet, "/mnt/rd/inference_lora/unet", adapter_name="identity1")
    adapters_weights1 = load_peft_weights("/mnt/rd/inference_lora/unet", device="cuda:0")
    pipe.unet.load_state_dict(adapters_weights1, strict = False)
    pipe.unet.to("cuda", torch.float16)

    
    negative_prompt = "low quality, blurry, unfinished, cartoon"
    images = pipe(
        ["sks person with a cat", "sks person with a cat", "sks person with a dog", "sks person"], 
        negative_prompt=[negative_prompt] * 4,
                  height=640, width=640, num_inference_steps=50, guidance_scale=2.5).images
    canvas = Image.new('RGB', (640*4, 640))
    fembs = []
    for i, image in enumerate(images):
        canvas.paste(image, (640*i, 0))
        fembs.append(get_face_embedding(image))

    main_comps = None
    cross_comps = None
    if face_emb1 is not None:
        face_similarity = []
        for im, femb in enumerate(fembs):
            if femb is not None:
                face_image = femb.flatten()#.chunk(2, dim=0)
                m_face_image = face_emb1.flatten()#.chunk(2, dim=0)
                # normalize both embeddings
                face_image = torch.nn.functional.normalize(face_image, p=2, dim=0)
                m_face_image = torch.nn.functional.normalize(m_face_image, p=2, dim=0)
                main_comp = (torch.nn.functional.cosine_similarity(face_image, m_face_image, dim=0))

                # also compare cross fembs
                cross_face_similarity = []
                for ib, femb2 in enumerate(fembs):
                    if femb2 is not None and ib != im:
                        face_image2 = femb2.flatten()#.chunk(2, dim=0)
                        face_image2 = torch.nn.functional.normalize(face_image2, p=2, dim=0)
                        cross_face_similarity.append(torch.nn.functional.cosine_similarity(face_image, face_image2, dim=0))

                face_similarity.append((main_comp,torch.Tensor(cross_face_similarity).mean()))

        if len(face_similarity) > 0:
            main_comps, cross_comps = zip(*face_similarity)
            main_comps = {i: main_comps[i] for i in range(len(main_comps))}
            cross_comps = {i: cross_comps[i] for i in range(len(cross_comps))}

    # Calculate the new dimensions
    new_width = canvas.width // 1
    new_height = canvas.height // 1

    # Resize the image
    canvas = canvas.resize((new_width, new_height))

    temporary_file_path = "/home/ubuntu/AutoLoRADiscovery/" + ("b_face.jpg" if fake_face_embedding is not None else ("b_latent.jpg" if custom_latent is not None else "ood_face.jpg") )
    canvas.save(temporary_file_path, format="JPEG", quality=20)  # quality=20 for higher compression

    # generate_denoising_sequence_viz(prev_latents, temporary_file_path.replace(".jpg", "_seq.jpg"))



    return  wandb.Image(temporary_file_path, 
    caption="[ "+(caption)+" ] "+(custom_title if custom_title is not None else "Generated Image")
    ), 0.0, (main_comps, cross_comps)
    # wandb.log({"image":)


def generate_lora_from_image(lora_diffusion, scheduler, file_name, lora_vae):
    global weight_dimensions
    # global pipe
    latents = torch.randn(1, 4096).cuda()#.to(torch.float16)
    face_emb1 = torch.Tensor(get_face_embedding(file_name)).unsqueeze(0)
    # print("Rendering from face embedding, shape:", face_emb1.shape)
    with torch.no_grad():
        for t in scheduler.timesteps:
            # Concatenate conditional and unconditional embeddings
            # combined_embeddings = torch.cat([face_emb1, uncond_face_emb], dim=0)
            
            # Duplicate latents for conditional and unconditional predictions
            # latent_model_input = torch.cat([latents] * 2)
            
            # Get both conditional and unconditional predictions
            noise_pred, pred_cond = lora_diffusion(
                latents, 
                t=t.unsqueeze(0).cuda().half(), 
                face_embeddings=face_emb1
            )
            
            # Separate conditional and unconditional predictions
            # noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            
            # Perform classifier-free guidance
            # cfg_scale = 3.0
            # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Update latents
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    latents = latents[:, :]
    # latents = latents #* 0.0152  #* std_w2w
    latents = w2w_to_lora(lora_vae, latents)  
    # latents = latents * 0.0152

    return latents, face_emb1

def generate_image_from_latents(lora_weights, prompt):
    global weight_dimensions
    if os.path.exists("/mnt/rd/inference_lora"):
        os.system("rm -rf /mnt/rd/inference_lora")
    
    latents = lora_weights#.unsqueeze(0)

    unflatten(latents.detach().clone(), weight_dimensions, "/mnt/rd/inference_lora")
    
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51", 
                                         torch_dtype=torch.float16,safety_checker = None,
                                         requires_safety_checker = False).to(device)

    
    pipe.unet = PeftModel.from_pretrained(pipe.unet, "/mnt/rd/inference_lora/unet", adapter_name="identity1")
    adapters_weights1 = load_peft_weights("/mnt/rd/inference_lora/unet", device="cuda:0")
    pipe.unet.load_state_dict(adapters_weights1, strict = False)
    pipe.unet.to("cuda", torch.float16)

    

    images = pipe([prompt] * 4, height=640, width=640, num_inference_steps=50, guidance_scale=3.0).images
    canvas = Image.new('RGB', (640*4, 640))
    fembs = []
    for i, image in enumerate(images):
        canvas.paste(image, (640*i, 0))
        fembs.append(get_face_embedding(image))

    # Calculate the new dimensions
    new_width = canvas.width // 1
    new_height = canvas.height // 1

    # Resize the image
    canvas = canvas.resize((new_width, new_height))

    temporary_file_path = "/home/ubuntu/AutoLoRADiscovery/" + "l_b_face.jpg"
    canvas.save(temporary_file_path, format="JPEG", quality=40)  # quality=20 for higher compression

    return  temporary_file_path
    # wandb.log({"image":)

def render_from_lora_weights(latents, file_name):
    if os.path.exists("/mnt/rd/inference_lora"):
        os.system("rm -rf /mnt/rd/inference_lora")
    
    latents = latents.unsqueeze(0)

    unflatten(latents.detach().clone(), weight_dimensions, "/mnt/rd/inference_lora")
    
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51", 
                                         torch_dtype=torch.float16,safety_checker = None,
                                         requires_safety_checker = False).to(device)

    
    pipe.unet = PeftModel.from_pretrained(pipe.unet, "/mnt/rd/inference_lora/unet", adapter_name="identity1")
    adapters_weights1 = load_peft_weights("/mnt/rd/inference_lora/unet", device="cuda:0")
    pipe.unet.load_state_dict(adapters_weights1, strict = False)
    pipe.unet.to("cuda", torch.float16)

    

    images = pipe(["A photo of a sks person"] * 4, height=640, width=640, num_inference_steps=50, guidance_scale=3.0).images
    canvas = Image.new('RGB', (640*4, 640))
    fembs = []
    for i, image in enumerate(images):
        canvas.paste(image, (640*i, 0))
        fembs.append(get_face_embedding(image))

  

    # Calculate the new dimensions
    new_width = canvas.width // 1
    new_height = canvas.height // 1

    # Resize the image
    canvas = canvas.resize((new_width, new_height))

    temporary_file_path = "/home/ubuntu/AutoLoRADiscovery/" + f"{file_name}.jpg"
    canvas.save(temporary_file_path, format="JPEG", quality=20)  # quality=20 for higher compression



    return  wandb.Image(temporary_file_path, 
    caption=file_name 
    )
    # wandb.log({"image":)
    