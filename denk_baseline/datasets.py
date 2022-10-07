import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torch.utils.data import Dataset

import pydicom
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut

from .utils import preprocess_image, preprocess_mask2onehot, preprocess_single_mask, get_img_names, rle2mask, resize_if_need_up, resize_if_need_down


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
    def __init__(self, images_dir, masks_dir, csv_path, stage, labels=None, img_w=None, img_h=None, augs=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels = labels if labels else [0, 1]
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        self.stage = stage

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

    def get_mixup(self, df):
        sub_df = df.sample(n=4, replace=True)

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
        
        if self.stage == 'train' and np.random.random() < 0.15:
            image, mask = self.get_mixup(self.df)
        else:
            img_name = '{}.tiff'.format(info['id'])
            img_path = os.path.join(self.images_dir, img_name)
            msk_path = os.path.join(self.masks_dir, img_name)
            image = cv2.imread(img_path)
            img_h, img_w = image.shape[:2]

            if self.stage == 'train' and np.random.random() < 0.5:
                image = cv2.resize(image, dsize=None, fx=0.064, fy=0.064, interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(image, dsize=(img_w, img_h), interpolation=cv2.INTER_LINEAR)
            
            if self.stage == 'train' and np.random.random() < 0.2:
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
            'mask': mask.unsqueeze(0).float(),
        }


class HubmapDatasetFromRLE(Dataset):
    def __init__(self, images_dir, csv_path, stage, labels=None, img_w=None, img_h=None, augs=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels if labels else [0, 1]
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        self.stage = stage

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

    def get_mixup(self, df):
        sub_df = df.sample(n=4, replace=True)

        img_ids = [row['id'] for i, row in sub_df.iterrows()]
        mask_rles = [row['rle'] for i, row in sub_df.iterrows()]
        np.random.shuffle(img_ids)

        half_h = self.img_h // 2
        half_w = self.img_w // 2

        r_image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        r_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        for i, img_id in enumerate(img_ids):
            img_name = f'{img_id}.jpg'
            img_path = os.path.join(self.images_dir, img_name)

            image = cv2.imread(img_path)
            img_h, img_w = image.shape[:2]
            mask = rle2mask(mask_rles[i], (img_h, img_w))

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
        
        if self.stage == 'train' and np.random.random() < 0.15:
            image, mask = self.get_mixup(self.df)
        else:
            img_name = '{}.jpg'.format(info['id'])
            img_path = os.path.join(self.images_dir, img_name)
            image = cv2.imread(img_path)
            img_h, img_w = image.shape[:2]

            if self.stage == 'train' and np.random.random() > 0.5:
                image = cv2.resize(image, dsize=None, fx=0.064, fy=0.064, interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(image, dsize=(img_w, img_h), interpolation=cv2.INTER_LINEAR)
            
            if self.stage == 'train' and np.random.random() < 0.2:
                normalizer = np.random.choice(self.normalizers) 
                image, _, _ = normalizer.normalize(I=image, stains=True)

            mask = rle2mask(info['rle'], (img_h, img_w))

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
            'mask': mask.unsqueeze(0).float(),
        }


def prune_image(im, mask, thr=0.990):
    for l in reversed(range(im.shape[1])):
        if (np.sum(mask[:, l]) / float(mask.shape[0])) > thr:
            im = np.delete(im, l, 1)
    for l in reversed(range(im.shape[0])):
        if (np.sum(mask[l, :]) / float(mask.shape[1])) > thr:
            im = np.delete(im, l, 0)
    return im


def mask_median(im, val=255):
    masks = [None] * 3
    for c in range(3):
        masks[c] = im[..., c] >= np.median(im[:, :, c]) - 5
    mask = np.logical_and(*masks)
    im[mask, :] = val
    return im, mask

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

        image = cv2.imread(img_path)
        # image = Image.open(img_path)
        # image.thumbnail((self.img_w, self.img_h))
        # image = np.array(image)

        # image, mask = mask_median(image)
        # image = prune_image(image, mask)

        # image = resize_if_need_up(image, max_h=self.min_img_h, max_w=self.min_img_w, interpolation=cv2.INTER_LINEAR)
        # image = resize_if_need_down(image, self.img_h, self.img_w, interpolation=cv2.INTER_AREA)

        if self.augs is not None:
            image = self.augs(image=image)['image']
            
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h, mean=mean, std=std)
        # label = 0.99 if self.df.iloc[index]['num_label'] else 0.0
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
        image = cv2.imread(img_path)

        if self.augs is not None:
            image = self.augs(image=image)['image']
            
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h, mean=mean, std=std)
        label = self.df.iloc[index]['num_label']

        return {
            'image': image, 
            'label': label,
            'oh_label': np.array([0, 1]) if label else np.array([1, 0]) 
        }
    
class RSNADataset(Dataset):
    def __init__(self, csv_path, img_w=None, img_h=None, augs=None):
        self.df = pd.read_csv(csv_path)
        self.img_w = img_w
        self.img_h = img_h
        self.augs = augs
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        dcm_path = self.df.iloc[index]['dcm_path']
        dcm_raw = pydicom.dcmread(dcm_path)
        dcm_image = apply_voi_lut(dcm_raw.pixel_array, dcm_raw)
        
        image = dcm_image - dcm_image.min()
        image /= image.max()
        image *= 255
        image = image.astype(np.uint8)

        if self.augs is not None:
            image = self.augs(image=image)['image']
            
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])

        image = preprocess_image(image, img_w=self.img_w, img_h=self.img_h, mean=mean, std=std)
        labels = self.df.iloc[index][[f'N{x}' for x in range(1, 9)]]

        return {
            'image': image, 
            'labels': labels.values.astype(np.int32),
        }
