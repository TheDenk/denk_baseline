import os
import glob
import importlib

import cv2
import torch
import numpy as np


def get_img_names(folder, img_format='png'):
    img_paths = glob.glob(os.path.join(folder, f'*.{img_format}'))
    img_names = [os.path.basename(x) for x in img_paths]
    return img_names

def preprocess_image(image, img_w=None, img_h=None):
    img = image.copy()
    if img_w and img_h:
        img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def preprocess_mask2onehot(image, labels, img_w=None, img_h=None):
    input_img = image.copy()
    if img_w and img_h:
        img = cv2.resize(input_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    img = np.array([(img == x) for x in labels])
    img = np.stack(img, axis=-1).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def preprocess_single_mask(image, labels, img_w=None, img_h=None):
    img = image.copy()
    if img_w and img_h:
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    for index, label in enumerate(labels):
        img[img == label] = index
    img = torch.from_numpy(img)
    return img

def process_img2np(image):
    img = image.cpu().clone()
    img = img.permute(1, 2, 0).numpy() * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def process_multimask2np(image, labels):
    img = image.cpu().clone()
    img = img.permute(1, 2, 0).numpy().astype(bool)
    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for c_index in range(c):
        mask[img[:, :, c_index]] = labels[c_index]
    return mask

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    return get_obj_from_str(config['target'])(**config.get('params', dict()))