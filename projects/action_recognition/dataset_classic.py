import os

import cv2
import torch
import numpy as np
import pandas as pd
from denku import read_video, resize_to_min_side
from torch.utils.data import Dataset


AUGMENTATIONS = {
    'zero': {
        'hflip': 0.0,
        'crop': 0.0,
        'rotate': 0.0, 
        'shear': 0.0, 
        'translate': 0.0, 
        'cutout': 0.0, 
        'skip': 0.0, 
        'blur': 0.0, 
        'invert': 0.0,
        'mixup': 0.0,
        'cutmix': 0.0,
    },
    'easy': {
        'hflip': 0.25,
        'crop': 0.25,
        'rotate': 0.25, 
        'shear': 0.25, 
        'translate': 0.25, 
        'cutout': 0.25, 
        'skip': 0.25, 
        'blur': 0.15, 
        'invert': 0.15,
        'mixup': 0.15,
        'cutmix': 0.25,
    },
    'medium': {
        'hflip': 0.5,
        'crop': 0.5,
        'rotate': 0.5, 
        'shear': 0.5, 
        'translate': 0.5, 
        'cutout': 0.5, 
        'skip': 0.5, 
        'blur': 0.25, 
        'invert': 0.25,
        'mixup': 0.5,
        'cutmix': 0.5,
    },
    'hard': {
        'hflip': 0.5,
        'crop': 0.9,
        'rotate': 0.9, 
        'shear': 0.9, 
        'translate': 0.9, 
        'cutout': 0.9, 
        'skip': 0.9, 
        'blur': 0.5, 
        'invert': 0.25,
        'mixup': 0.5,
        'cutmix': 0.9,
    },
}


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Crop the center of a numpy array image to target dimensions.
    
    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        target_h: Target height (must be <= image height)
        target_w: Target width (must be <= image width)
        
    Returns:
        Cropped center image as numpy array
    """
    h, w = img.shape[:2]
    
    if target_h > h or target_w > w:
        raise ValueError(
            f"Target dimensions ({target_h}, {target_w}) must be smaller than "
            f"image dimensions ({h}, {w})"
        )
    
    start_y = h // 2 - target_h // 2
    start_x = w // 2 - target_w // 2
    
    return img[start_y:start_y + target_h, start_x:start_x + target_w]
    

def calculate_cross_coefs(frames_count):
    cross_mid = 0.5 + np.random.random() * 0.1 * np.sign(np.random.random() - 0.5)
    cross_window  = 0.25 + np.random.random() * 0.2 * np.sign(np.random.random() - 0.5)
    
    start_cross = np.clip(cross_mid - cross_window, 0.0, 1.0)
    end_cross = np.clip(cross_mid + cross_window, 0.0, 1.0)
    
    start_cross_frame, end_cross_frame = max(1, int(frames_count * start_cross)), min(int(frames_count * end_cross), frames_count - 1)
    cross_frame_count = end_cross_frame - start_cross_frame
    
    cross_coefs = np.linspace(0, 1, cross_frame_count, dtype=np.float32) 
    first_video_coefs = np.zeros(start_cross_frame)
    second_video_coefs = np.ones(frames_count - end_cross_frame)
    coefs = np.concatenate([first_video_coefs, cross_coefs, second_video_coefs])[:, None, None, None]
    return coefs, cross_mid


class ActionsDataset(Dataset):
    def __init__(
        self,
        txt_file_path: str,
        data_folder: str,
        target_side: int,
        sample_n_frames: int,
        augs_type: str = "zero",
        is_train: bool = True,
    ) -> None:
        df = pd.read_csv(
            txt_file_path, delimiter=' ', names=['path', 'class']
        )
        self.is_train = is_train
        if not self.is_train:
            df = df.groupby('class').sample(n=185, random_state=42)

        df["tool"] = df["path"].apply(lambda x: x.split('/')[0])
        df = df[df["tool"] != "hydraulic-jack"]
        
        self.video_names = df['path'].apply(
            lambda image_name: os.path.join(data_folder, image_name),
        ).to_numpy()
        
        self.length = len(self.video_names)
        self.tools = df["tool"].values
        self.label_indexes = df['class'].to_numpy()
        self.target_side = target_side
        self.sample_n_frames = sample_n_frames
        self.augs_proba = AUGMENTATIONS[augs_type]
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def get_default_item(self, idx):
        label = self.label_indexes[idx]
        video_path = self.video_names[idx]
        tool = self.tools[idx]
        
        stride = 3
        if self.is_train:
            stride = np.random.choice([1, 2, 3])   
             
        video, _ = read_video(
            video_path, 
            start_frame=0, 
            frames_count=self.sample_n_frames,
            # max_side=self.target_side,
            frame_stride=stride
        )
        return video, label

    def apply_augs(self, pixel_values):
        ## F, C, H, W
        ## random center crop
        if np.random.random() < self.augs_proba['crop']:
            crop_pad = np.random.choice([8, 16, 32, 48])
            side = self.target_side + crop_pad
            # print(pixel_values)
            pixel_values = [resize_to_min_side(img, side) for img in pixel_values]
            pixel_values = [center_crop(img, self.target_side, self.target_side) for img in pixel_values]
        else:
            pixel_values = [resize_to_min_side(img, self.target_side) for img in pixel_values]

        ## random hflip
        if np.random.random() < self.augs_proba['hflip']:
            pixel_values = [img[:, ::-1] for img in pixel_values]
            
        h, w, c = pixel_values[0].shape
        ## rotate
        if np.random.random() < self.augs_proba['rotate']:
            degrees = [-15, 15]
            angle = np.random.randint(degrees[0], degrees[1])
            pixel_values = [rotate_image(img, angle) for img in pixel_values]
            
        ## shear
        if np.random.random() < self.augs_proba['shear']:
            shear = [-0.05, 0.05]
            x_shear = np.random.uniform(shear[0], shear[1])
            y_shear = np.random.uniform(shear[0], shear[1])
            
            shear_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
            pixel_values = [cv2.warpAffine(img, shear_mat, (w, h)) for img in pixel_values]
        
        ## translate
        if np.random.random() < self.augs_proba['translate']:
            value = min(h, w) // 20
            translate = [-value, value]
            x_move = np.random.randint(translate[0], translate[1])
            y_move = np.random.randint(translate[0], translate[1])
            
            transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
            pixel_values = [cv2.warpAffine(img, transform_mat, (w, h)) for img in pixel_values]
        
        ## cutout
        if np.random.random() < self.augs_proba['cutout']:
            num_cuts = np.random.randint(16, 32)
            pixel_values = np.stack(pixel_values)
            
            for _ in range(num_cuts):
                for frame_index in range(pixel_values.shape[0]):
                    cut_w = np.random.randint(w // 30, w // 10)
                    cut_h = np.random.randint(h // 30, h // 10)

                    cut_x = np.random.randint(0, w - cut_w)
                    cut_y = np.random.randint(0, h - cut_h)
                    pixel_values[frame_index, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w, :] = 0
            
        ## random skip frame
        if np.random.random() < self.augs_proba['skip']:
            num_skips = np.random.randint(1, self.sample_n_frames // 3)
            skip_indexes = [np.random.randint(0, self.sample_n_frames) for _ in range(num_skips)]
            for frame_index in skip_indexes:
                pixel_values[frame_index] = np.zeros((h, w, c), dtype=np.uint8)
                
        ## blur
        if np.random.random() < self.augs_proba['blur']:
            pixel_values = [cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT) for img in pixel_values]
            
        ## invert
        if np.random.random() < self.augs_proba['invert']:
            pixel_values = [np.invert(img) for img in pixel_values]
        
        if isinstance(pixel_values, list):
            pixel_values = np.stack(pixel_values)
        return pixel_values

    def get_augmented_item(self, idx):
        pixel_values, label = self.get_default_item(idx)

        pixel_values = self.apply_augs(pixel_values)
        pixel_values = (pixel_values.astype(np.float32) / 255.0 - self.mean) / self.std
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()

        oh_label = torch.zeros(7)
        oh_label[label] = 1
        sample = dict(image=pixel_values, label=label, oh_label=oh_label)
        return sample
        
    def __getitem__(self, idx):
        item = self.get_augmented_item(idx)
        mixup_done = False
        
        if np.random.random() < self.augs_proba['mixup']:
            mixup_done = True
            mixup_index = np.random.randint(0, self.length)
            mixup_item = self.get_augmented_item(mixup_index)

            mixup_alpha = np.random.randint(50, 400) / 1000.0
            item['image'] = mixup_item['image'] * mixup_alpha + item['image'] * (1.0 - mixup_alpha)
            item['oh_label'] = mixup_item['oh_label'] * mixup_alpha + item['oh_label'] * (1.0 - mixup_alpha)

        if (not mixup_done) and (np.random.random() < self.augs_proba['cutmix']):
            cutmix_coefs, cutmix_alpha = calculate_cross_coefs(self.sample_n_frames)
            cutmix_index = np.random.randint(0, self.length)
            cutmix_item = self.get_augmented_item(cutmix_index)

            item['image'] = item['image'] * cutmix_coefs + cutmix_item['image'] * (1.0 - cutmix_coefs)
            item['oh_label'] = item['oh_label'] * cutmix_alpha + cutmix_item['oh_label'] * (1.0 - cutmix_alpha)
            
        return item
        
    def __len__(self) -> int:
        return self.length


class ActionsMultiheadDataset(ActionsDataset):        
    def get_default_item(self, idx):
        label = self.label_indexes[idx]
        video_path = self.video_names[idx]
        tool = self.tools[idx]
        
        stride = 3
        if self.is_train:
            stride = np.random.choice([1, 2, 3])   
             
        video, _ = read_video(
            video_path, 
            start_frame=0, 
            frames_count=self.sample_n_frames,
            # max_side=self.target_side,
            frame_stride=stride
        )
        return video, label

    def get_augmented_item(self, idx):
        pixel_values, label = self.get_default_item(idx)

        pixel_values = self.apply_augs(pixel_values)
        pixel_values = (pixel_values.astype(np.float32) / 255.0 - self.mean) / self.std
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()

        oh_label = torch.zeros(self.sample_n_frames, 7)
        oh_label[:, label] = 1
        sample = dict(image=pixel_values, label=label, oh_label=oh_label)
        return sample
        
    def __getitem__(self, idx):
        item = self.get_augmented_item(idx)
        mixup_done = False
        cutmix_done = False
        
        if np.random.random() < self.augs_proba['mixup']:
            mixup_done = True
            mixup_index = np.random.randint(0, self.length)
            mixup_item = self.get_augmented_item(mixup_index)

            mixup_alpha = np.random.randint(50, 400) / 1000.0
            item['image'] = mixup_item['image'] * mixup_alpha + item['image'] * (1.0 - mixup_alpha)
            item['oh_label'] = mixup_item['oh_label'] * mixup_alpha + item['oh_label'] * (1.0 - mixup_alpha)

        if (not mixup_done) and (np.random.random() < self.augs_proba['cutmix']):
            cutmix_done = True
            cutmix_index = np.random.randint(0, self.length)
            cutmix_item = self.get_augmented_item(cutmix_index)
            
            cutmix_coefs, cutmix_alpha = calculate_cross_coefs(self.sample_n_frames)
            item['image'] = item['image'] * cutmix_coefs + cutmix_item['image'] * (1.0 - cutmix_coefs)
            item['oh_label'] = item['oh_label'] * cutmix_coefs[:, :, 0, 0] + cutmix_item['oh_label'] * (1.0 - cutmix_coefs[:, :, 0, 0])
            
        return item
        
    def __len__(self) -> int:
        return self.length
# params = dict(
#     txt_file_path="/home/raid/astashkin/action_recognition/data/v6/train.txt", 
#     data_folder="/home/raid/astashkin/action_recognition/data/clips",  
#     target_side=512,
#     sample_n_frames=32,
#     augs_type="hard",
# )
# action_dataset = ActionsDataset(**params)
# print(len(action_dataset))

# index = np.random.randint(0, len(action_dataset))
# item = action_dataset[index]


# # from tqdm import tqdm
# # for index in tqdm(range(len(action_dataset)), total=len(action_dataset)):
# #     item = action_dataset[index]
    
# item = action_dataset[0]
# images = item['image']
# images = images.permute(0, 2, 3, 1)
# images = images  * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
# images = images.float().numpy()
# images = np.clip(images * 255, 0, 255).astype(np.uint8)


# print(f"LABEL: {item['label']} | OH LABEL: {item['oh_label']}")
# show_images(images[:4], figsize=(15, 15))