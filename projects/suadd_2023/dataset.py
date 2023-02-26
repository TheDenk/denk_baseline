import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from denk_baseline.utils import read_image, preprocess_image, preprocess_mask2onehot


class SUADDDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_path, stage, img_w=None, img_h=None, augs=None, mosaic_proba=0.0):
        self.df = pd.read_csv(csv_path).reset_index()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.stage = stage
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        self.mosaic_proba = mosaic_proba
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255]

    def __len__(self):
        return self.df.shape[0]
    
    def get_mixup(self):
        sub_df = self.df.sample(n=4, replace=True)

        img_names = sub_df['image_name'].values
        np.random.shuffle(img_names)

        half_h = self.img_h // 2
        half_w = self.img_w // 2

        r_image = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        r_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.images_dir, img_name)
            msk_path = os.path.join(self.masks_dir, img_name)
            image = read_image(img_path, to_rgb=False, flag=cv2.IMREAD_GRAYSCALE)
            mask = read_image(msk_path, to_rgb=False, flag=cv2.IMREAD_GRAYSCALE)
            
            if self.augs is not None:
                item = self.augs(image=image, mask=mask)
                image = item['image']
                mask = item['mask']
            
            image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)


            if i == 0:
                r_image[0:half_h, 0:half_w] = image[0:half_h, 0:half_w].copy()
                r_mask[0:half_h, 0:half_w] = mask[0:half_h, 0:half_w].copy()
            elif i == 1:
                r_image[0:half_h, half_w:] = image[0:half_h, half_w:].copy()
                r_mask[0:half_h, half_w:] = mask[0:half_h, half_w:].copy()
            elif i == 2:
                r_image[half_h:, 0:half_w] = image[half_h:, 0:half_w].copy()
                r_mask[half_h:, 0:half_w] = mask[half_h:, 0:half_w].copy()
            elif i == 3:
                r_image[half_h:, half_w:] = image[half_h:, half_w:].copy()
                r_mask[half_h:, half_w:] = mask[half_h:, half_w:].copy()
        
        return r_image, r_mask
    
    def get_item(self, index):
        image_name = self.df.iloc[index]['image_name']
        img_path = os.path.join(self.images_dir, image_name)
        msk_path = os.path.join(self.masks_dir, image_name)
        
        image = read_image(img_path, to_rgb=False, flag=cv2.IMREAD_GRAYSCALE)
        mask = read_image(msk_path, to_rgb=False, flag=cv2.IMREAD_GRAYSCALE)
        
        if self.augs is not None:
            item = self.augs(image=image, mask=mask)
            image = item['image']
            mask = item['mask']
        
        return image, mask
    
    def __getitem__(self, index):
        if self.stage == 'train' and np.random.random() < 0.5:
            image, mask = self.get_mixup()
        else:
            image, mask = self.get_item(index)
        
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])
        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h, mean=mean, std=std)
        mask = preprocess_mask2onehot(mask, self.labels, to_torch=True, img_w=self.img_w, img_h=self.img_h)
        
        return {
            'image': image, 
            'mask': mask,
        }