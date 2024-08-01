import gradio as gr
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import diffusers
import clip
import numpy as np
import cv2
from common.loras import patch_lora
from common.utils import make_weight_vector, recover_lora, convert_to_multi, rand_merge_layerwise
from discover_lora_diffusion.models import LoraDiffusion
import diffusers
import torch
from tqdm import tqdm 
from common.render import generate_lora_from_image, generate_image_from_latents
from common.utils import make_weight_vector, recover_lora, convert_to_multi, rand_merge_layerwise


lora_diffusion = LoraDiffusion(
    # data_dim=1_365_504, 
    # model_dim=512, 
    # ff_mult=3, 
    # chunks=1, 
    # act=torch.nn.SiLU, 
    #  num_blocks=6,
    # layers_per_block=3
        # data_dim=1_365_504,
        # model_dim=768,
        # ff_mult=3,
        # chunks=1,
        # act=torch.nn.SiLU,
        # num_blocks=6,
        # layers_per_block=4
        data_dim=4096,
        model_dim=1024,
        ff_mult=2,
        chunks=1,
        act=torch.nn.SiLU,
        num_blocks=4,
        layers_per_block=6,
)

config = {
    "_class_name": "UnCLIPScheduler",
    "_diffusers_version": "0.17.0.dev0",
    "clip_sample": True,
    "clip_sample_range": 10.0,  # Adjusted to match your data range
    "num_train_timesteps": 200,  # Reduced from 1000
    "prediction_type": "sample",
    "variance_type": "fixed_small_log"
}

scheduler = diffusers.UnCLIPScheduler.from_config(config)
state_dict = torch.load('/mnt/rd/diff_lora_mdl/checkpoint-clpface_newdataset22_small_50step_denoise_reluflat-4000')#'/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/diffusion_lora/checkpoint-clpface_muladainx_risid_lin_mob_llr_swp-48000')#'/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/diffusion_lora/checkpoint-clpface_muladainx_risid_lin_mob_swp-4000')#/home/ubuntu/AutoLoRADiscovery/discover_lora_diffusion/diffusion_lora/checkpoint-clpface_self_moattn_muladainx_risid_lin_mob_swp_46000')
lora_diffusion.load_state_dict(state_dict)
lora_diffusion = lora_diffusion.cuda().eval()



from discover_lora_vae.models import LoraVAE
# lora_vae = None
lora_vae = LoraVAE(
    input_dim=99_648,
    latent_dim=4096,
).cuda()

lora_vae.load_state_dict(torch.load("/mnt/rd/model-out-2/checkpoint-37000", map_location="cuda"))

# Function to generate image using LoRA and prompt



loras_map = {

}
# Function to process images and return an ID
def process_images(filename):
    print ("generating lora for ", filename)
    id = hash(filename)
    if id in loras_map:
        print ("already processed")
        return id
    _lora_lantent, _pred_faceemb = generate_lora_from_image(lora_diffusion, scheduler, filename, lora_vae)


    # hash the filename and store the _lora_lantent and _pred_faceemb
    
    loras_map[id] = (_lora_lantent, _pred_faceemb)

    print ("done generating lora for ", filename)
    return id

# Function to handle prompts using the ID
def handle_prompt(id, prompt):
    lora_latent, _ = loras_map[id]
    img_file = generate_image_from_latents(lora_latent, prompt)
    print ("done generating image for ", id, prompt, img_file)
    return  img_file

# Gradio interface setup
with gr.Blocks(title="AutoLoRADiscovery") as demo:
    gr.Markdown("## AutoLoRADiscovery")
    gr.Markdown("This demo is for converting your face pictures into a LoRA that's applied into a Stable Diffusion model.")
    gr.Markdown("Made as a display of work done by [Ethan Smith](https://github.com/ethansmith2000/AutoLoRADiscovery).")

    # File input component for multiple images, displaying selected images
    file_input = gr.Image(label="Upload your own face pictures", type="filepath")

    # Prompt input component
    prompt_input = gr.Textbox(label="Enter Prompt", interactive=False)
    
    # Output gallery component
    output_gallery = gr.Image(label="Generated View", height=400) 
    
    # Function to process images and handle prompt
    def process_and_prompt(files, prompt):
        id = process_images(files)
        return handle_prompt(id, prompt)
    
    # Button to submit prompt, initially disabled
    submit_btn = gr.Button("Submit Prompt", interactive=False)

    # Function to enable prompt input once files are uploaded
    def enable_prompt(files):
        if files:
            return gr.update(interactive=True)
        return gr.update(interactive=False)
    
    # Function to enable submit button once both inputs are filled
    def enable_submit(files, prompt):
        if files and prompt:
            return gr.update(interactive=True)
        return gr.update(interactive=False)

    # Set up event handlers
    file_input.change(enable_prompt, inputs=file_input, outputs=prompt_input)
    file_input.change(enable_submit, inputs=[file_input, prompt_input], outputs=submit_btn)
    prompt_input.change(enable_submit, inputs=[file_input, prompt_input], outputs=submit_btn)
    
    # Set up submit button click event
    submit_btn.click(process_and_prompt, [file_input, prompt_input], output_gallery)

# Launch the Gradio interface
demo.launch(share=True)
