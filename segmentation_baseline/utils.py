import os
import glob
import json
import importlib

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_img_names(folder, img_format='png'):
    img_paths = glob.glob(os.path.join(folder, f'*.{img_format}'))
    img_names = [os.path.basename(x) for x in img_paths]
    return img_names

def preprocess_image(image, img_w=None, img_h=None, mean=np.array([0, 0, 0]), std=np.array([1, 1, 1])):
    img = image.copy()
    if img_w and img_h:
        img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ((img.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)
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
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def show_image(image, figsize=(5, 5), cmap=None, title='', xlabel=None, ylabel=None, axis=False):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.show();
    
def show_images(images, figsize=(5, 5), title='', cmap=None, xlabel=None, ylabel=None, axis=False):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis(axis)
    plt.show();
    
def mask2rle(img):
    '''https://www.kaggle.com/paulorzp/rle-functions-run-length-encode-decode'''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape):
    '''https://www.kaggle.com/paulorzp/rle-functions-run-length-encode-decode'''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape).T

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def normalized_image_to_numpy(image, mean=np.array([0, 0, 0]), std=np.array([1, 1, 1])):
    '''
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    '''
    img = image.cpu()
    img = inverse_normalize(img, mean=mean, std=std)
    img = img.permute(1, 2, 0).numpy()*255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def make_img_padding(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    max_h = max(img_h, max_h)
    max_w = max(img_w, max_w)
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1, y1 = 0, 0
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg

def split_on_chunks(data, n_chunks):
    chunk_size = int(len(data) / n_chunks)
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

def split_on_patches(image, patch_h, patch_w, step_x=None, step_y=None):
    img_h, img_w = image.shape[:2]
    
    patches = []
    step_y = step_y if step_y else patch_h
    step_x = step_x if step_x else patch_w
    
    for y in range(0, img_h - (patch_h - step_y), step_y):
        for x in range(0, img_w - (patch_w - step_x), step_x):
            patches.append(image[y:y + patch_h, x:x + patch_w])
    
    return np.array(patches)

def glue_image_patches(patches, patch_h, patch_w, img_h, img_w, img_c, step_x=None, step_y=None):
    if img_c:
        bg = np.zeros((img_h, img_w, img_c), dtype=np.uint8)
    else:
        bg = np.zeros((img_h, img_w), dtype=np.uint8)
        
    p_index = 0
    step_y = step_y if step_y else patch_h
    step_x = step_x if step_x else patch_w
    
    for y in range(0, img_h - (patch_h - step_y), step_y):
        for x in range(0, img_w - (patch_w - step_x), step_x):
            bg[y:y + patch_h, x:x + patch_w] = patches[p_index]
            p_index += 1
            
    return bg

def glue_mask_patches(patches, patch_h, patch_w, img_h, img_w, img_c, step_x=None, step_y=None):
    if img_c:
        bg = np.zeros((img_h, img_w, img_c), dtype=np.float32)
    else:
        bg = np.zeros((img_h, img_w), dtype=np.float32)
        
    p_index = 0
    step_y = step_y if step_y else patch_h
    step_x = step_x if step_x else patch_w
    
    over_h = patch_h - step_y
    over_w = patch_w - step_x
    
    for y in range(0, img_h - (patch_h - step_y), step_y):
        for x in range(0, img_w - (patch_w - step_x), step_x):
            
            bg[y:y + patch_h, x:x + patch_w] += patches[p_index]
            p_index += 1
            
            if over_w and x > 0:
                bg[y:y + patch_h, x:x + over_w] /= 2
                
            if over_h and y > 0:
                bg[y:y + over_h, x:x + patch_w] /= 2
    
    return bg
        

# class PatchTrainDataset(Dataset):
#     def __init__(self, images_names, images_folder, masks_folder, patch_h=None, patch_w=None, max_h=None, max_w=None, step_x=None, step_y=None, augmentations=None):
#         self.images_folder = images_folder
#         self.masks_folder = masks_folder
#         self.images_names = images_names
#         self.augmentations = augmentations

#     def __len__(self):
#         return len(self.images_names)
    
#     def __getitem__(self, index):
#         img_name = self.images_names[index]
#         img_path = os.path.join(self.images_folder, img_name)
#         msk_path = os.path.join(self.masks_folder, img_name)
        
#         image = cv2.imread(img_path)
#         mask = cv2.imread(msk_path, 0)
        
#         orig_h, orig_w = image.shape[:2]
        
#         if self.augmentations is not None:
#             item = self.augmentations(image=image, mask=mask)
#             image = item['image']
#             mask = item['mask']
        
#         image = make_img_padding(image, max_h, max_w)
#         mask = make_img_padding(mask[:, :, None], max_h, max_w)[:, :, 0]
        
#         img_patches = split_on_patches(
#             image, 
#             patch_h, 
#             patch_w,
#             step_x=step_x,
#             step_y=step_y,
#         )
            
#         msk_patches = split_on_patches(
#             mask, 
#             patch_h, 
#             patch_w,
#             step_x=step_x,
#             step_y=step_y,
#         )
        
#         img_patches = [preprocess_image(img_patch, resize=False) for img_patch in img_patches]
#         sg_msk_patches = [preprocess_mask(g_item, resize=False) for g_item in msk_patches]
#         mt_msk_patches = [preprocess_multy_mask(g_item, resize=False) for g_item in msk_patches]
        
#         return {
#             'img_patches': img_patches, 
#             'sg_msk_patches': sg_msk_patches, 
#             'mt_msk_patches': mt_msk_patches,
#             'orig_h': orig_h,
#             'orig_w': orig_w,
#         }
    
    
# class PatchTestDataset(Dataset):
#     def __init__(self, images_names, images_folder, patch_h=None, patch_w=None, max_h=None, max_w=None, step_x=None, step_y=None, augmentations=None):
#         self.images_folder = images_folder
#         self.images_names = images_names
#         self.augmentations = augmentations

#     def __len__(self):
#         return len(self.images_names)
    
#     def __getitem__(self, index):
#         img_name = self.images_names[index]
#         img_path = os.path.join(self.images_folder, img_name)
        
#         image = cv2.imread(img_path)
#         orig_h, orig_w = image.shape[:2]
        
#         if self.augmentations is not None:
#             item = self.augmentations(image=image)
#             image = item['image']
        
#         image = make_img_padding(image, max_h, max_w)
        
#         img_patches = split_on_patches(
#             image, 
#             patch_h, 
#             patch_w,
#             step_x=step_x,
#             step_y=step_y,
#         )
        
#         img_patches = [preprocess_image(img_patch, resize=False) for img_patch in img_patches]
        
#         return {
#             'img_patches': img_patches,
#             'image_name': img_name,
#             'orig_h': orig_h,
#             'orig_w': orig_w,
#         }
    
# train_dataset = PatchTrainDataset(TRAIN_IMG_NAMES, TRAIN_IMAGES_FOLDER, TRAIN_MASKS_FOLDER)
# item = train_dataset[512]
# orig_h, orig_w = item['orig_h'], item['orig_w']

# img = glue_image_patches(
#     [process_img2np(x) for x in item['img_patches']], 
#     GLOBAL_CONFIG['PATCH_H'], 
#     GLOBAL_CONFIG['PATCH_W'],
#     GLOBAL_CONFIG['MAX_H'], 
#     GLOBAL_CONFIG['MAX_W'],
#     img_c=3,
#     step_x=GLOBAL_CONFIG['step_x'],
#     step_y=GLOBAL_CONFIG['step_y'],
# )[:orig_h, :orig_w]

# msk = glue_mask_patches(
#     [process_multimask2np(x, to_image=False) for x in item['mt_msk_patches']], 
#     GLOBAL_CONFIG['PATCH_H'], 
#     GLOBAL_CONFIG['PATCH_W'],
#     GLOBAL_CONFIG['MAX_H'], 
#     GLOBAL_CONFIG['MAX_W'],
#     img_c=len(LABELS),
#     step_x=GLOBAL_CONFIG['step_x'],
#     step_y=GLOBAL_CONFIG['step_y'],
# )[:orig_h, :orig_w]

# msk = mask_one_hot_to_image(msk.astype(bool))

# show_image_mask(img, msk*10, figsize=(20, 30))