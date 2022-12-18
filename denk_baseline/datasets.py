import os

import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .utils import read_image, preprocess_image, preprocess_mask2onehot, preprocess_single_mask, get_img_names


class SegmentationMulticlassDataset(Dataset):
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

        image = read_image(img_path)
        mask = read_image(msk_path, flag=cv2.IMREAD_GRAYSCALE)

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


class SegmentationBinaryDataset(Dataset):
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

        image = read_image(img_path)
        mask = read_image(msk_path, flag=cv2.IMREAD_GRAYSCALE)

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

class ClassificationBinaryDataset(Dataset):
    def __init__(self, csv_path, images_dir, stage, img_w=None, img_h=None, augs=None, img_format='jpg'):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        self.stage = stage
        self.img_format = img_format
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_name = self.df.iloc[index]['image_id']
        img_name = f'{img_name}.{self.img_format }' if self.img_format is not None else img_name
        img_path = os.path.join(self.images_dir, img_name)

        image = read_image(img_path)
        if self.augs is not None:
            image = self.augs(image=image)['image']

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h)
        label = self.df.iloc[index]['num_label']
        return {
            'image': image, 
            'label': label,
        }


class ClassificationMulticlassDataset(Dataset):
    def __init__(self, csv_path, images_dir, stage, img_w=None, img_h=None, augs=None, img_format='jpg'):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        self.stage = stage
        self.img_format = img_format
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_name = self.df.iloc[index]['image_id']
        img_name = f'{img_name}.{self.img_format }' if self.img_format is not None else img_name
        img_path = os.path.join(self.images_dir, img_name)
        image = read_image(img_path)

        if self.augs is not None:
            image = self.augs(image=image)['image']

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h)
        label = self.df.iloc[index]['num_label']

        return {
            'image': image, 
            'label': label,
            'oh_label': np.array([0, 1]) if label else np.array([1, 0]) 
        }


