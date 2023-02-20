import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from denk_baseline.utils import read_image, preprocess_image


class RSNADataset(Dataset):
    def __init__(self, csv_path, images_dir, stage, img_w=None, img_h=None, augs=None, mixup_proba=0.0, roi_proba=0.0, clahe_proba=0.0):
        self.df = pd.read_csv(csv_path).reset_index()[:-2]
        # if stage == 'train':
        #     self.df = self.df[self.df['is_additional'] == False]
        self.images_dir = images_dir
        self.stage = stage
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        self.canser_ids = self.df[self.df.cancer == 1].index
        self.mixup_proba = mixup_proba
        self.roi_proba = roi_proba
        self.clahe_proba = clahe_proba

    def __len__(self):
        return self.df.shape[0]

    def clahe(self, in_img):
        img_0 = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
        clip_1 = np.random.random()
        clip_2 = clip_1 + np.random.random()
        img_1 = cv2.createCLAHE(clipLimit=clip_1).apply(img_0)
        img_2 = cv2.createCLAHE(clipLimit=clip_2).apply(img_0)
        img_out = cv2.merge((img_0, img_1, img_2))
        return img_out

    def img2roi(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_img = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        ys = contour.squeeze()[:, 0]
        xs = contour.squeeze()[:, 1]
        roi =  img[np.min(xs):np.max(xs), np.min(ys):np.max(ys)]
        return roi
    
    def get_item(self, index):
        img_id = self.df.iloc[index]['image_id']
        patient_id = self.df.iloc[index]['patient_id']
        if self.stage == 'train':
            images_dir = '/media/user/FastNVME/archive/processed_images' if self.df.iloc[index]['is_additional'] else self.images_dir
        else:
            images_dir = self.images_dir
        img_path = os.path.join(images_dir, f'{patient_id}', f'{img_id}.png')
        
        image = read_image(img_path).astype(np.uint8)
        
        if np.random.random() < self.roi_proba:
            image = self.img2roi(image)
        
        if np.random.random() < self.clahe_proba:
            image = self.clahe(image)

        if self.augs is not None:
            image = self.augs(image=image)['image']
            
        label = self.df.iloc[index]['cancer']
        image = cv2.resize(image, (self.img_w, self.img_h))
        return image, label
    
    def __getitem__(self, index):
        image, label = self.get_item(index)
        
        if np.random.random() < self.mixup_proba:
            c_index = np.random.randint(0, len(self.canser_ids))
            s_id = self.canser_ids[c_index]
            # s_id = np.random.randint(0, self.df.shape[0])

            s_image, s_label = self.get_item(s_id)
            alpha = min(max(np.random.random(), 0.1), 0.9)
            beta = 1.0 - alpha
            image = cv2.addWeighted(image, alpha, s_image, beta, 0.0)
            label = label*alpha + s_label*beta
        
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])
        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h, mean=mean, std=std)
        
        return {
            'image': image, 
            'label': label,
            # 'oh_label': torch.tensor([1 - label, 0 + label])
        }