import gc
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from contextlib import nullcontext
import sys



            
sam_models = {
    'vit_b': 'ckpts/sam_vit_b.pth',
    'vit_l': 'ckpts/sam_vit_l.pth',
    'vit_h': 'ckpts/sam_vit_h.pth'
}

def get_sam_predictor(model_type='vit_h', device=None, image=None):

    sam = sam_model_registry[model_type](checkpoint=sam_models[model_type])
    sam = sam.to(device)

    predictor = SamPredictor(sam)
    if image is not None:
        predictor.set_image(image)
    return predictor


def run_sam(predictor: SamPredictor, selected_points):

    if len(selected_points) == 0:
        return []
    input_points = [p for p, _ in selected_points]
    input_labels = [int(l) for _, l in selected_points]

    masks, _, _ = predictor.predict(
                        point_coords = np.array(input_points),
                        point_labels = input_labels,
                        multimask_output = False, # single object output
    )
    visible_mask = 255 * np.squeeze(masks).astype(np.uint8) # (256, 256)
    overlay_mask = [(masks,'visible_mask')]
    
    return visible_mask, overlay_mask


def process_input(input_im):
    normalized_image = torch.from_numpy(input_im).float().permute(2, 0, 1) / 255. # [0, 255] to [0, 1]
    normalized_image = normalized_image * 2 - 1 # [0, 1] to [-1, 1]
    return normalized_image.unsqueeze(0)
