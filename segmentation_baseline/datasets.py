import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .utils import preprocess_image, preprocess_mask2onehot, preprocess_single_mask, get_img_names


class MulticlassDataset(Dataset):
    def __init__(self, images_dir, masks_dir, labels, img_w=None, img_h=None, augs=None, img_format='png'):
        self.img_names = get_img_names(images_dir, img_format=img_format)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels = labels
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.images_dir, img_name)
        msk_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(msk_path, 0)

        if self.augs is not None:
            item = self.augs(image=image, mask=mask)
            image = item['image']
            mask = item['mask']

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h)
        oh_mask = preprocess_mask2onehot(mask, self.labels, img_w=self.img_w, img_h=self.img_h)
        sg_mask = preprocess_single_mask(mask, self.labels, img_w=self.img_w, img_h=self.img_h)

        return {
            'image': image, 
            'oh_mask': oh_mask, 
            'sg_mask': sg_mask,
        }


class BinaryDataset(Dataset):
    def __init__(self, images_dir, masks_dir, labels=None, img_w=None, img_h=None, augs=None, img_format='png'):
        self.img_names = get_img_names(images_dir, img_format=img_format)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels = labels if labels else [0, 1]
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.images_dir, img_name)
        msk_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(msk_path, 0)

        if self.augs is not None:
            item = self.augs(image=image, mask=mask)
            image = item['image']
            mask = item['mask']

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h)
        mask = preprocess_single_mask(mask, self.labels, img_w=self.img_w, img_h=self.img_h)
        
        return {
            'image': image, 
            'mask': mask.unsqueeze(0),
        }


class HubmapDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_path, labels=None, img_w=None, img_h=None, augs=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels = labels if labels else [0, 1]
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs

        self.normalizers = self.get_stain_normalizers()
        
    def __len__(self):
        return self.df.shape[0]

    def get_stain_normalizers(self):
        import torchstain as ts
        targets = [
            cv2.imread('./stain_images/1.jpg'),
            cv2.imread('./stain_images/2.jpg'),
            cv2.imread('./stain_images/3.jpg'),
            cv2.imread('./stain_images/4.jpg'),
        ] 
        normalizers = []
        for target in targets:
            normalizer = ts.MacenkoNormalizer(backend='numpy')
            normalizer.fit(target)
            normalizers.append(normalizer)
        return normalizers

    def get_mixup(self, df, organ):
        sub_df = df[df['organ'] == organ].sample(n=4, replace=True)

        img_ids = [row['id'] for i, row in sub_df.iterrows()]
        np.random.shuffle(img_ids)

        half_h = self.img_h // 2
        half_w = self.img_w // 2

        r_image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        r_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        for i, img_id in enumerate(img_ids):
            img_name = f'{img_id}.tiff'
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            if np.random.random() > 0.5:
                img_h, img_w = image.shape[:2]
                image = cv2.resize(image, dsize=None, fx=0.064, fy=0.064, interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

            if np.random.random() > 0.5:
                normalizer = np.random.choice(self.normalizers) 
                image, _, _ = normalizer.normalize(I=image, stains=True)

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

    def __getitem__(self, index):
        info = self.df.iloc[index]
        
        if np.random.random() < 0.2:
            organ = info['organ']
            image, mask = self.get_mixup(self.df, organ)
        else:
            img_name = '{}.tiff'.format(info['id'])
            img_path = os.path.join(self.images_dir, img_name)
            msk_path = os.path.join(self.masks_dir, img_name)
            image = cv2.imread(img_path)
            img_h, img_w = image.shape[:2]

            if np.random.random() > 0.5:
                image = cv2.resize(image, dsize=None, fx=0.064, fy=0.064, interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(image, dsize=(img_w, img_h), interpolation=cv2.INTER_LINEAR)
            
            if np.random.random() > 0.66:
                normalizer = np.random.choice(self.normalizers) 
                image, _, _ = normalizer.normalize(I=image, stains=True)

            mask = cv2.imread(msk_path, 0)

            if self.augs is not None:
                item = self.augs(image=image, mask=mask)
                image = item['image']
                mask = item['mask']

        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h, mean=mean, std=std)
        mask = preprocess_single_mask(mask, self.labels, img_w=self.img_w, img_h=self.img_h)

        return {
            'image': image, 
            'mask': mask.unsqueeze(0),
        }