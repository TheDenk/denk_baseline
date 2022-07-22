import os

import cv2
from torch.utils.data import Dataset

from .utils import preprocess_image, preprocess_mask2onehot, preprocess_single_mask, get_img_names


class TrainDataset(Dataset):
    def __init__(self, images_dir, masks_dir, labels, img_w=256, img_h=256, augs=None, img_format='png'):
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
