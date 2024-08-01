import torch
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import glob
# import clip
import tqdm
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align
import numpy as np
import cv2
import os
import random
from p_tqdm import p_map
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

def rtn_face_get(self, img, face):
    aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    #print(cv2.imwrite("aimg.png", aimg))
    face.embedding = self.get_feat(aimg).flatten()
    face.crop_face = aimg
    return face.embedding

ArcFaceONNX.get = rtn_face_get
app = FaceAnalysis(name="buffalo_sc", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def crop_face_box(cv2_img, bbox, face_ratio=1.0):
    h, w, _ = cv2_img.shape
    box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if box_w < box_h:
        diff = box_h - box_w
        bbox[0] -= diff // 2
        bbox[2] += diff // 2
    else:
        diff = box_w - box_h
        bbox[1] -= diff // 2
        bbox[3] += diff // 2
    min_dist = min(bbox[0], w - bbox[2], bbox[1], h - bbox[3]) * face_ratio - 1
    bbox = [int(bbox[0] - min_dist), int(bbox[1] - min_dist), int(bbox[2] + min_dist), int(bbox[3] + min_dist)]
    bbox = [max(0, coord) for coord in bbox]  # Ensure coordinates are not negative

    face_image = cv2_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return face_image

def crop_face(image):
    cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = app.get(cv2_img)
    if len(faces) == 0:
        return None, None
    bbox = faces[0].bbox.astype(int)
    cropped = crop_face_box(cv2_img, bbox)
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), (faces[0].embedding).flatten()

ids = torch.load("/mnt/rd/identity_df.pt")

# find all items in column "file"
indexes = [str(k) for k in ids.index.values.tolist()]

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
# dino_model = AutoModel.from_pretrained('facebook/dinov2-large')
# dino_model = dino_model.cuda()
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

print("Clip model loaded")


@torch.no_grad()
def process_batch(batch_images):
    # print("batch_images.shape=", batch_images.shape)
    # emb = dino_model(batch_images.cuda()).last_hidden_state.mean(dim=1)
    emb = clip_model.encode_image(batch_images.cuda())
    return emb.cpu()

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc= self.images[idx]
        img = Image.open(img_loc).convert('RGB')
        cropped_img, face_emb = crop_face(img)
        if cropped_img is None:
            return None
        # full_img = preprocess(img).squeeze(0)
        # face_img = preprocess(cropped_img, return_tensors="pt").pixel_values.squeeze(0)
        return face_emb, img_loc
print("number of indexes", len(indexes))
celebA_filenames = []
for indx in indexes:
    fl_loc = f"/mnt/rd/img_align_celeba/{indx}"
    if os.path.exists(fl_loc):
        celebA_filenames.append(f"/mnt/rd/img_align_celeba/{indx}")
# celebA_filenames = [f for f in celebA_filenames if f.split("/")[-1] in indexes]
print("filtered celebA_filenames", len(celebA_filenames))

img_dataset = ImgDataset(celebA_filenames)
def ignore_none_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
img_data_loader = torch.utils.data.DataLoader(img_dataset, batch_size=256, num_workers=24, collate_fn=ignore_none_collate)
# Main processing loop
batch_size = 128  # Adjust based on GPU memory
all_embeddings = {}


for  face_embeddings, image_locs  in tqdm.tqdm(img_data_loader, desc="Extracting sir clip"):
    file_names = [f.split("/")[-1] for f in image_locs]
    # print("full_images.shape=", full_images.shape)
    # batch_images = full_images#torch.cat([, face_images], dim=0)
    # with torch.no_grad():
    #     full_embeddings = process_batch(batch_images)
    # full_embeddings, face_embeddings = embeddings.chunk(2, dim=0)
    for file_name, face_ebms in zip(file_names, face_embeddings):
        all_embeddings[file_name] = face_ebms#[full_embs, ]

# save
torch.save(all_embeddings, "celeba_arc_embs.pt")