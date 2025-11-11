import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path

from src.utils.train_util import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=4, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        image_dir='rendering_zero123plus',
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir

        with open(os.path.join(root_dir, meta_fname)) as f:
            lvis_dict = json.load(f)
        paths = []
        for k in lvis_dict.keys():
            paths.append(lvis_dict[k])
        self.paths = paths
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[-16:]            
        else:
            self.paths = self.paths[:-16]
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    
    
    def process_numpy_image(self, image, color):
        color = np.array(color, dtype=np.float32)
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            image = image[:, :, :3]
            image = image * alpha[:, :, np.newaxis] + color * (1 - alpha[:, :, np.newaxis])
        else:
            alpha = np.ones(image.shape[:2], dtype=np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).unsqueeze(0).contiguous().float()

        return image    
    
    
    def load_im(self, path, color):
        pil_img = Image.open(path)
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        
        image = self.process_numpy_image(image, color)
        return image
    
    def load_mv(self, path, color):
        pil_img = Image.open(path)
        side_len = pil_img.width//2
        subimages = [pil_img.crop((x, y, x + side_len, y+side_len)) for y in range(0, pil_img.height, side_len) for x in range(0, pil_img.width, side_len)]
        
        image_arrays = [np.asarray(img, dtype=np.float32) / 255. for img in subimages]
        
        rgb_arrays = [self.process_numpy_image(img, color) for img in image_arrays]
        return rgb_arrays
    
    def __getitem__(self, index):
        while True:
            image_path = os.path.join(self.root_dir, self.image_dir, self.paths[index])
            cond_path = os.path.join(image_path, 'occluded.png')
            mv_path = os.path.join(image_path, 'mv.png')
            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []

            cond_img = self.load_im(cond_path, bkg_color)
            rgb_arrays = self.load_mv(mv_path, bkg_color)

            img_list.extend(rgb_arrays)
            break
        
        imgs = torch.stack(img_list, dim=0).float()

        data = {
            'cond_imgs': cond_img,           # (3, H, W)
            'target_imgs': imgs,        # (6, 3, H, W)
        }
        return data
