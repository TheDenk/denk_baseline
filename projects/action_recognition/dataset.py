import os
from copy import deepcopy

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
        'rotate_temporal': 0.0, 
        'shear': 0.0, 
        'pepper': 0.0,
        'salt': 0.0,
        'translate': 0.0, 
        'cutout': 0.0, 
        'skip_frames': 0.0, 
        'gaussian_blur': 0.0, 
        'invert': 0.0,
        'mixup': 0.0,
        'cutmix': 0.0,
    },
    'easy': {
        'hflip': 0.25,
        'crop': 0.25,
        'rotate_temporal': 0.25, 
        'shear': 0.25, 
        'pepper': 0.25,
        'salt': 0.25,
        'translate': 0.25, 
        'cutout': 0.25, 
        'skip_frames': 0.25, 
        'gaussian_blur': 0.15, 
        'invert': 0.15,
        'mixup': 0.15,
        'cutmix': 0.15,
    },
    'medium': {
        'hflip': 0.5,
        'crop': 0.5,
        'rotate_temporal': 0.5, 
        'shear': 0.5, 
        'pepper': 0.5,
        'salt': 0.5,
        'translate': 0.5, 
        'cutout': 0.5, 
        'skip_frames': 0.5, 
        'gaussian_blur': 0.25, 
        'invert': 0.25,
        'mixup': 0.25,
        'cutmix': 0.25,
    },
    'hard': {
        'hflip': 0.5,
        'crop': 0.9,
        'rotate_temporal': 0.9, 
        'shear': 0.9, 
        'pepper': 0.9,
        'salt': 0.9,
        'translate': 0.9, 
        'cutout': 0.9, 
        'skip_frames': 0.9, 
        'gaussian_blur': 0.5, 
        'invert': 0.5,
        'mixup': 0.5,
        'cutmix': 0.5,
    },
    'test': {
        'hflip': 0.5,  # 1, 6
        'crop': 0.9,  # 1, 6
        'rotate_temporal': 0.5,  # 2, 6
        'shear': 0.25, # 6 
        'pepper': 0.1,  # 1
        'salt': 0.1,  # 1
        'translate': 0.25, # 6
        'cutout': 0.9,  # 2
        'skip_frames': 0.75,  # 2, 6
        'gaussian_blur': 0.5, # 3, 6
        'invert': 0.5, # 3, 6
        'mixup': 0.5, # 3, 4, 5, 6
        'cutmix': 0.5, # 3, 4, 5, 6
    },
}

def set_random_pixel_values(image: np.ndarray, alpha: float, value:int = 0) -> np.ndarray:
    """
    Apply salt augmentation to a numpy image by adding white pixels randomly.
    
    Args:
        image: Input image as a numpy array (H, W) or (H, W, C).
        alpha: Percentage of pixels to replace with value (0.0 to 1.0).
        valeu: Pixel value (For example: 255 - salt augmentation, 0 - paper augmentation)
    
    Returns:
        Augmented image as numpy array with same shape as input.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")

    changed_image = image.copy()
    total_pixels = changed_image.shape[0] * changed_image.shape[1]
    num_changed_pixels = int(alpha * total_pixels)
    
    if changed_image.ndim == 2: 
        coords = [np.random.randint(0, i, num_changed_pixels) for i in changed_image.shape]
        changed_image[coords[0], coords[1]] = value
    else: 
        coords = [np.random.randint(0, i, num_changed_pixels) for i in changed_image.shape[:2]]
        changed_image[coords[0], coords[1]] = [value] * changed_image.shape[2]
    return changed_image


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
            f'Target dimensions ({target_h}, {target_w}) must be smaller than '
            f'image dimensions ({h}, {w})'
        )

    start_y = h // 2 - target_h // 2
    start_x = w // 2 - target_w // 2

    return img[start_y:start_y + target_h, start_x:start_x + target_w]


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by a specified angle around its center.

    Args:
        image (np.ndarray): Input image of shape (H, W, C) or (H, W) with values in range [0, 255]
        angle (float): Rotation angle in degrees. Positive values rotate counter-clockwise.

    Returns:
        np.ndarray: Rotated image with the same shape and data type as the input image

    Note:
        The rotation is performed around the center of the image.
        The output image maintains the same dimensions as the input image,
        with black padding added where necessary.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


class Sequential:
    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.random_order = random_order

    def __call__(self, video):
        if self.random_order:
            shuffled_transforms = deepcopy(self.transforms)
            np.random.shuffle(shuffled_transforms)
            for transform in shuffled_transforms:
                video = transform(video)
        else:
            for transform in self.transforms:
                video = transform(video)
        return video

class OneOf:
    def __init__(self, p, transforms):
        self.p = p
        self.transforms = transforms

    def __call__(self, video):
        if np.random.random() < self.p:
            transform = np.random.choice(self.transforms)
            video = transform(video)
        return video

class BaseVideoAugmentator:
    def __init__(self, p):
        self.p = p

    def apply_augmentation(self, video):
        raise NotImplementedError()

    def __call__(self, video):
        if np.random.random() < self.p:
            video = self.apply_augmentation(video)
        return video

class HorizontalFlip(BaseVideoAugmentator):
    def apply_augmentation(self, video):
        return [img[:, ::-1] for img in video]

class Invert(BaseVideoAugmentator):
    def apply_augmentation(self, video):
        return [np.invert(img) for img in video]
        
class GaussianBlur(BaseVideoAugmentator):
    def __init__(self, p, kernel_size):
        self.p = p
        self.kernel_size = kernel_size
    
    def apply_augmentation(self, video):
        return [cv2.GaussianBlur(img, self.kernel_size, cv2.BORDER_DEFAULT) for img in video]

class Salt(BaseVideoAugmentator):
    def __init__(self, p, alpha_range):
        self.p = p
        self.alpha_range = alpha_range
    
    def apply_augmentation(self, video):
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        return [set_random_pixel_values(img, alpha=alpha, value=255) for img in video]

class Pepper(BaseVideoAugmentator):
    def __init__(self, p, alpha_range):
        self.p = p
        self.alpha_range = alpha_range
    
    def apply_augmentation(self, video):
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        return [set_random_pixel_values(img, alpha=alpha, value=0) for img in video]
        
class ResizeToMinSide(BaseVideoAugmentator):
    def __init__(self, p, target_side):
        self.p = p
        self.target_side = target_side

    def apply_augmentation(self, video):
        return [resize_to_min_side(img, self.target_side) for img in video]

class CenterCrop(BaseVideoAugmentator):
    def __init__(self, p, target_side):
        self.p = p
        self.target_side = target_side

    def apply_augmentation(self, video):
        img_h, img_w = video[0].shape[:2]
        video = [center_crop(img, self.target_side, self.target_side) for img in video]
        return [cv2.resize(img, (img_w, img_h), cv2.INTER_LINEAR) for img in video]
        
class RotateConstant(BaseVideoAugmentator):
    def __init__(self, p, degrees_range):
        self.p = p
        self.degrees_range = degrees_range

    def apply_augmentation(self, video):
        angle = np.random.randint(self.degrees_range[0], self.degrees_range[1])
        return [rotate_image(img, angle) for img in video]

class RotateTemporal(BaseVideoAugmentator):
    def __init__(self, p, degrees_range):
        self.p = p
        self.degrees_range = degrees_range
        self.center_of_degrees = degrees_range[0] + (degrees_range[1] - degrees_range[0]) // 2

    def apply_augmentation(self, video):
        frames_count = len(video)
        start_angle = np.random.randint(self.degrees_range[0], self.center_of_degrees)
        end_angle = np.random.randint(self.center_of_degrees, self.degrees_range[1])
        angles = np.linspace(start_angle, end_angle, frames_count, dtype=np.int32)
        return [rotate_image(img, angle) for img, angle in zip(video, angles)]

class Shear(BaseVideoAugmentator):
    def __init__(self, p, share_range, use_constant=False):
        self.p = p
        self.share_range = share_range
        self.use_constant = use_constant
    
    def apply_augmentation(self, video):
        img_h, img_w = video[0].shape[:2]
        shear_matrices = []
        if self.use_constant:
            x_shear = np.random.uniform(self.share_range[0], self.share_range[1])
            y_shear = np.random.uniform(self.share_range[0], self.share_range[1])
            shear_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
            shear_matrices.extend([shear_mat] * len(video))
        else:
            for _ in range(len(video)):
                x_shear = np.random.uniform(self.share_range[0], self.share_range[1])
                y_shear = np.random.uniform(self.share_range[0], self.share_range[1])
                shear_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
                shear_matrices.append(shear_mat)
        video = [cv2.warpAffine(img, shear_mat, (img_w, img_h)) for img, shear_mat in zip(video, shear_matrices)]
        return video

class Translate(BaseVideoAugmentator):
    def __init__(self, p, translate_range, use_constant=False):
        self.p = p
        self.translate_range = translate_range
        self.use_constant = use_constant
    
    def apply_augmentation(self, video):
        img_h, img_w = video[0].shape[:2]
        transform_matrices = []
        if self.use_constant:
            x_move = np.random.randint(self.translate_range[0], self.translate_range[1])
            y_move = np.random.randint(self.translate_range[0], self.translate_range[1])
            transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
            transform_matrices.extend([transform_matrices] * len(video))
        else:
            for _ in range(len(video)):
                x_move = np.random.randint(self.translate_range[0], self.translate_range[1])
                y_move = np.random.randint(self.translate_range[0], self.translate_range[1])
                transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
                transform_matrices.append(transform_mat)
                
        video = [cv2.warpAffine(img, transform_mat, (img_w, img_h)) for img, transform_mat in zip(video, transform_matrices)]
        return video

class SkipFrames(BaseVideoAugmentator):
    def __init__(self, p, skip_frames_range):
        self.p = p
        self.skip_frames_range = skip_frames_range
    
    def apply_augmentation(self, video):
        frames_count = len(video)
        num_skips = np.random.randint(*self.skip_frames_range)
        skip_indexes = [np.random.randint(0, frames_count) for _ in range(num_skips)]
        for frame_index in skip_indexes:
            video[frame_index] = np.zeros_like(video[frame_index])
        return video
        
class Cutout(BaseVideoAugmentator):
    def __init__(self, p, num_cuts_range, cut_width_range, cut_height_range):
        self.p = p
        self.num_cuts_range = num_cuts_range
        self.cut_width_range = cut_width_range
        self.cut_height_range = cut_height_range
    
    def apply_augmentation(self, video):
        img_h, img_w = video[0].shape[:2]
        num_cuts = np.random.randint(*self.num_cuts_range)
        for _ in range(num_cuts):
            for frame_index in range(len(video)):
                cut_w = np.random.randint(*self.cut_width_range)
                cut_h = np.random.randint(*self.cut_height_range)

                cut_x = np.random.randint(0, img_w - cut_w)
                cut_y = np.random.randint(0, img_h - cut_h)
                video[frame_index][cut_y:cut_y + cut_h, cut_x:cut_x + cut_w, :] = 0
        return video
    

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


def get_augs(aug_type, random_order=False):
    tresholds = AUGMENTATIONS[aug_type]
    augmentations = Sequential([
        Salt(p=tresholds['salt'], alpha_range=(0.05, 0.15)),
        Pepper(p=tresholds['pepper'], alpha_range=(0.05, 0.15)),

        OneOf(p=tresholds['crop'], transforms=[
            CenterCrop(p=1.0, target_side=256 - 8),
            CenterCrop(p=1.0, target_side=256 - 16),
            CenterCrop(p=1.0, target_side=256 - 32),
        ]),
        HorizontalFlip(p=tresholds['hflip']),
        RotateTemporal(p=tresholds['rotate_temporal'], degrees_range=[-15, 15]),
        Shear(p=tresholds['shear'], share_range=(-0.1, 0.1), use_constant=False),
        Translate(p=tresholds['translate'], translate_range=(-50, 50), use_constant=False),
        Cutout(
            p=tresholds['cutout'], 
            num_cuts_range=(16, 32), 
            cut_width_range=(5, 25), 
            cut_height_range=(5, 25)
        ),
        SkipFrames(p=tresholds['skip_frames'], skip_frames_range=(1, 10)),
        GaussianBlur(p=tresholds['gaussian_blur'], kernel_size=(3, 3)),
        Invert(p=tresholds['invert']),
    ], random_order=random_order)
    return augmentations


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
        self.augmentations = get_augs(augs_type, random_order=False)
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
        pixel_values = [cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) for img in pixel_values]
        pixel_values = self.augmentations(pixel_values)
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
        sample = dict(inputs=pixel_values, label=label, oh_label=oh_label)
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
            item['inputs'] = mixup_item['inputs'] * mixup_alpha + item['inputs'] * (1.0 - mixup_alpha)
            item['oh_label'] = mixup_item['oh_label'] * mixup_alpha + item['oh_label'] * (1.0 - mixup_alpha)

        if (not mixup_done) and (np.random.random() < self.augs_proba['cutmix']):
            cutmix_done = True
            cutmix_index = np.random.randint(0, self.length)
            cutmix_item = self.get_augmented_item(cutmix_index)
            
            cutmix_coefs, cutmix_alpha = calculate_cross_coefs(self.sample_n_frames)
            item['inputs'] = item['inputs'] * cutmix_coefs + cutmix_item['inputs'] * (1.0 - cutmix_coefs)
            item['oh_label'] = item['oh_label'] * cutmix_alpha + cutmix_item['oh_label'] * (1.0 - cutmix_alpha)
            
        return item
        
    def __len__(self) -> int:
        return self.length
    


class ActionsMultiheadDataset(Dataset):
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
        self.augmentations = get_augs(augs_type)
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
            max_side=self.target_side,
            frame_stride=stride
        )
        return video, label
    
    def apply_augs(self, pixel_values):
        ## F, C, H, W
        pixel_values = self.augmentations(pixel_values)
        if isinstance(pixel_values, list):
            pixel_values = np.stack(pixel_values)
        return pixel_values

    def get_augmented_item(self, idx):
        pixel_values, label = self.get_default_item(idx)
        # print(pixel_values[0].shape)
        pixel_values = self.apply_augs(pixel_values)
        pixel_values = (pixel_values.astype(np.float32) / 255.0 - self.mean) / self.std
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()

        oh_label = torch.zeros(self.sample_n_frames, 7)
        oh_label[:, label] = 1
        sample = dict(inputs=pixel_values, label=label, oh_label=oh_label)
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
            item['inputs'] = mixup_item['inputs'] * mixup_alpha + item['inputs'] * (1.0 - mixup_alpha)
            item['oh_label'] = mixup_item['oh_label'] * mixup_alpha + item['oh_label'] * (1.0 - mixup_alpha)

        if (not mixup_done) and (np.random.random() < self.augs_proba['cutmix']):
            cutmix_done = True
            cutmix_index = np.random.randint(0, self.length)
            cutmix_item = self.get_augmented_item(cutmix_index)
            
            cutmix_coefs, cutmix_alpha = calculate_cross_coefs(self.sample_n_frames)
            item['inputs'] = item['inputs'] * cutmix_coefs + cutmix_item['inputs'] * (1.0 - cutmix_coefs)
            # print(cutmix_coefs[:, :, 0, 0])
            item['oh_label'] = item['oh_label'] * cutmix_coefs[:, :, 0, 0] + cutmix_item['oh_label'] * (1.0 - cutmix_coefs[:, :, 0, 0])
            
        return item
        
    def __len__(self) -> int:
        return self.length