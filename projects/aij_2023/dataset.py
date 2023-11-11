import os
import json
import random

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from decord import VideoReader
from pytorch_wavelets import DWTForward

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


AUGMENTATIONS = {
    'zero': {
        'crop': 0.0,
        'rotate': 0.0, 
        'shear': 0.0, 
        'translate': 0.0, 
        'cutout': 0.0, 
        'skip': 0.0, 
        'blur': 0.0, 
        'invert': 0.0,
        'mixup': 0.0,
    },
    'easy': {
        'crop': 0.25,
        'rotate': 0.25, 
        'shear': 0.25, 
        'translate': 0.25, 
        'cutout': 0.25, 
        'skip': 0.25, 
        'blur': 0.15, 
        'invert': 0.15,
        'mixup': 0.15,
    },
    'medium': {
        'crop': 0.5,
        'rotate': 0.5, 
        'shear': 0.5, 
        'translate': 0.5, 
        'cutout': 0.5, 
        'skip': 0.5, 
        'blur': 0.25, 
        'invert': 0.25,
        'mixup': 0.25,
    },
    'hard': {
        'crop': 0.9,
        'rotate': 0.9, 
        'shear': 0.9, 
        'translate': 0.9, 
        'cutout': 0.9, 
        'skip': 0.9, 
        'blur': 0.3, 
        'invert': 0.3,
        'mixup': 0.3,
    },
}

class EncoderDWT:
    def __init__(self):
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')

    def __call__(self, x):
        x = x.unsqueeze(0)
        freq = self.img_to_dwt(x)
        return freq.squeeze(0)

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq
    

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def resize_proportional(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(
        img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img


class TrainVideoDataset(Dataset):
    def __init__(self, 
                 csv_path, 
                 json_path, 
                 video_folder, 
                 min_side=256,
                 sample_size=224, 
                 sample_stride=2, 
                 sample_n_frames=16,
                 start_delta=8,
                 augs_proba='zero',
                 use_dwt=False,
                 *args, **kwargs
            ):
        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.start_delta = start_delta
        self.min_side = min_side
        self.augs_proba = AUGMENTATIONS[augs_proba]
        
        self.name2label = load_json(json_path)
        self.label2name = {v:k for k, v in self.name2label.items()}

        self.df = pd.read_csv(csv_path)

        test_df = pd.read_csv('/home/user/Projects/denk_baseline/projects/aij_2023/data/df_test.csv')
        self.df = pd.concat([self.df, test_df])
        # alphabet = list('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        # self.df = self.df[self.df['text'].isin(alphabet)]

        self.encoder_dwt = EncoderDWT() if use_dwt else None
        
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop((sample_size, sample_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
        ])

        self.a_transforms = A.Compose([
            A.LongestMaxSize(max_size=360, interpolation=1),
            A.PadIfNeeded(min_height=300, min_width=300, border_mode=0, value=(0,0,0)),
            A.CenterCrop(224, 224, always_apply=True, p=1.0)
        ], p=1.0)
        
    def __len__(self):
        return self.df.shape[0]
        
    def get_whole_action_event_bach(self, index):
        row = self.df.iloc[index]
        label = row.label #self.name2label[row.text]
        
        video_path = os.path.join(self.video_folder, f'{row.attachment_id}.mp4')
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        start_index = row.begin + np.random.randint(-self.start_delta, self.start_delta)
        start_index = max(0, start_index)
        
        end_index = row.end + np.random.randint(-self.start_delta, self.start_delta)
        end_index = min(video_length, end_index)
        
        batch_index = np.linspace(start_index, end_index, self.sample_n_frames, dtype=int)
        pixel_values = video_reader.get_batch(batch_index).asnumpy()
        del video_reader
        return pixel_values, label
    
    def get_event_batch(self, index, row=None):
        row = self.df.iloc[index] if row is None else row
        label = row.label #self.name2label[row.text]
        
        video_path = os.path.join(self.video_folder, f'{row.attachment_id}.mp4')
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        start_shift = np.random.randint(-self.start_delta, self.start_delta // 2)
        start_index = max(0, row.begin + start_shift)

        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1) - start_index
        
        batch_index = np.linspace(start_index, start_index + clip_length - 1, self.sample_n_frames, dtype=int)

        pixel_values = video_reader.get_batch(batch_index).asnumpy()
        del video_reader
        return pixel_values, label
    
    def get_no_event_batch(self, index):
        row = self.df.iloc[index]
        label = 1000 #self.name2label['no_event']
        
        video_path = os.path.join(self.video_folder, f'{row.attachment_id}.mp4')
        video_reader = VideoReader(video_path)
        min_frames_count = (self.sample_n_frames - 1) * self.sample_stride + 1
        
        if np.random.random() > 0.5: # no event from start
            clip_length = min(row.begin, min_frames_count)
            
            if clip_length == 0:
                raise ValueError('Get zero from Start')
            
            if row.begin - clip_length > 0:
                start_index = np.random.randint(0, row.begin - clip_length)
            else:
                start_index = 0
        else: # no event from end
            clip_length = min(row.length - row.end, min_frames_count)
            
            if clip_length == 0:
                raise ValueError('Get zero from End')
            
            if row.length - clip_length > row.end:
                start_index = np.random.randint(row.end, row.length - clip_length)
            else:
                start_index = row.end

        batch_index = np.linspace(start_index, start_index + clip_length - 1, self.sample_n_frames, dtype=int)
        pixel_values = video_reader.get_batch(batch_index).asnumpy()
            
        del video_reader
        return pixel_values, label
    
    def apply_augs(self, pixel_values):
        ## F, C, H, W
        ## random center crop
        if np.random.random() < self.augs_proba['crop']:
            side = np.random.choice([360, 320, 288])
            pixel_values = [resize_proportional(img, side, side) for img in pixel_values]
        else:
            pixel_values = [resize_proportional(img, self.min_side, self.min_side) for img in pixel_values]

        h, w, c = pixel_values[0].shape
        ## rotate
        if np.random.random() < self.augs_proba['rotate']:
            degrees = [-15, 15]
            angle = np.random.randint(degrees[0], degrees[1])
            pixel_values = [rotate_image(img, angle) for img in pixel_values]
            
        ## shear
        if np.random.random() < self.augs_proba['shear']:
            shear = [-0.05, 0.05]
            x_shear = random.uniform(shear[0], shear[1])
            y_shear = random.uniform(shear[0], shear[1])
            
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
                    cut_w = np.random.randint(w // 30, w // 20)
                    cut_h = np.random.randint(h // 30, h // 20)

                    cut_x = np.random.randint(0, w - cut_w)
                    cut_y = np.random.randint(0, h - cut_h)
                    pixel_values[frame_index, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w, :] = 0
            
        ## random skip frame
        if np.random.random() < self.augs_proba['skip']:
            num_skips = np.random.randint(1, 5)
            skip_indexes = [np.random.randint(0, self.sample_n_frames) for _ in range(num_skips)]
            for frame_index in skip_indexes:
                pixel_values[frame_index] = np.zeros((h, w, c), dtype=np.uint8)
                
        ## blur
        if np.random.random() < self.augs_proba['blur']:
            pixel_values = [cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT) for img in pixel_values]
            
        ## invert
        if np.random.random() < self.augs_proba['invert']:
            pixel_values = [np.invert(img) for img in pixel_values]
        
        if isinstance(pixel_values, list):
            pixel_values = np.stack(pixel_values)
        return pixel_values
    
    def get_item(self, index, row=None):
        while True:
            try:
                random_num = np.random.random()
                if random_num < 0.5 or row is not None:
                    pixel_values, label = self.get_event_batch(index, row)
                elif random_num < 0.99:
                    pixel_values, label = self.get_whole_action_event_bach(index)
                else:
                    pixel_values, label = self.get_no_event_batch(index)
                break
            except Exception as e:
                # print(index, e)
                index = np.random.randint(0, self.df.shape[0])
        
        pixel_values = self.apply_augs(pixel_values)
        pixel_values = np.stack([self.a_transforms(image=x)['image'] for x in pixel_values])
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0
        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values = pixel_values.permute(1, 0, 2, 3)
        
        oh_label = torch.zeros(1001)
        oh_label[label] = 1
        sample = dict(image=pixel_values, label=label, oh_label=oh_label)
        return sample

    def __getitem__(self, index):
        sample = self.get_item(index)

        ## mixup
        mixup_applied = False
        if np.random.random() < self.augs_proba['mixup']:
            mixup_index = np.random.randint(0, self.df.shape[0])
            mixup_sample = self.get_item(mixup_index)

            mixup_coef = np.random.randint(50, 400) / 1000.0
            sample['image'] = mixup_sample['image'] * mixup_coef + sample['image'] * (1.0 - mixup_coef)
            sample['oh_label'] = mixup_sample['oh_label'] * mixup_coef + sample['oh_label'] * (1.0 - mixup_coef)
            mixup_applied = True

        ## smoothed one class video
        if not mixup_applied and np.random.random() < self.augs_proba['mixup'] and sample['label'] != 1000:
            cross_row = self.df[self.df['label'] == sample['label']].sample(n=1, random_state=17).iloc[0]
            cross_sample = self.get_item(0, row=cross_row)
            
            cross_params = [np.random.normal(loc=0.5, scale=0.5) for _ in range(2)]
            cross_mid, cross_window = [np.clip(x, 0.0, 1.0) for x in cross_params]
            
            start_cross = np.clip(cross_mid - cross_window - 0.05, 0.0, 1.0)
            end_cross = np.clip(cross_mid + cross_window + 0.05, 0.0, 1.0)
            
            original_coefs = np.linspace(start_cross, end_cross, self.sample_n_frames, dtype=np.float32)[None, :, None, None]
            reversed_coefs = (1.0 - original_coefs)
            sample['image'] = sample['image'] * original_coefs + cross_sample['image'] * reversed_coefs

        return sample
    
    
# params = {
#     'tsv_path': '../../../datasets/slovo/annotations/SLOVO_DATAFRAME.tsv', 
#     'json_path': '../../../datasets/slovo/annotations/classes.json',
#     'video_folder': '../../../datasets/slovo/video', 
#     'sample_size': 224, 
#     'sample_stride': 2, 
#     'sample_n_frames': 32,
#     'start_delta': 8,
# }
# dataset = TrainVideoDataset(**params)

class ValidVideoDataset(Dataset):
    def __init__(self, 
                 csv_path, 
                 json_path, 
                 video_folder, 
                 min_side=256,
                 sample_size=224, 
                 sample_stride=2, 
                 sample_n_frames=16,
                 use_dwt=False,
                 *args, **kwargs
            ):
        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.name2label = load_json(json_path)
        self.df = pd.read_csv(csv_path)
        # alphabet = list('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        # self.df = self.df[self.df['text'].isin(alphabet)]

        self.encoder_dwt = EncoderDWT() if use_dwt else None
        
        self.pixel_transforms = transforms.Compose([
            # transforms.Resize(min_side, antialias=False),
            # transforms.CenterCrop((sample_size, sample_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
        ])
        self.a_transforms = A.Compose([
            A.LongestMaxSize(max_size=360, interpolation=1),
            A.PadIfNeeded(min_height=300, min_width=300, border_mode=0, value=(0,0,0)),
            A.CenterCrop(224, 224, always_apply=True, p=1.0)
        ], p=1.0)
    
    def get_batch(self, index):
        row = self.df.iloc[index]
        # print(row.keys())
        label = row.label #self.name2label[row.text]
        
        video_path = os.path.join(self.video_folder, f'{row.attachment_id}.mp4')
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        clip_length = min(video_length - row.begin, (self.sample_n_frames - 1) * self.sample_stride + 1)
        batch_index = np.linspace(row.begin, row.begin + clip_length - 1, self.sample_n_frames, dtype=int)
        pixel_values = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = np.stack([self.a_transforms(image=x)['image'] for x in pixel_values])
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        del video_reader
        return pixel_values, label
    
    def get_whole_action_event_bach(self, index):
        row = self.df.iloc[index]
        label = row.label #self.name2label[row.text]
        
        video_path = os.path.join(self.video_folder, f'{row.attachment_id}.mp4')
        video_reader = VideoReader(video_path)
        
        batch_index = np.linspace(row.begin, row.end, self.sample_n_frames, dtype=int)
        pixel_values = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = np.stack([self.a_transforms(image=x)['image'] for x in pixel_values])
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        del video_reader
        return pixel_values, label
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        while True:
            try:
                pixel_values, label = self.get_batch(index)
                break
            except Exception as e:
                print(index, e)
                index = np.random.randint(0, self.df.shape[0])
        
        pixel_values = pixel_values / 255.0
        pixel_values = self.pixel_transforms(pixel_values)

        if self.encoder_dwt is not None:
            pixel_values = [self.encoder_dwt(x) for x in pixel_values]
            pixel_values = torch.stack(pixel_values)
            
        pixel_values = pixel_values.permute(1, 0, 2, 3)

        oh_label = torch.zeros(1001)
        oh_label[label] = 1
        sample = dict(image=pixel_values, label=label, oh_label=oh_label)
        return sample
    
    
# params = {
#     'tsv_path': '../../../datasets/slovo/annotations/SLOVO_DATAFRAME.tsv', 
#     'json_path': '../../../datasets/slovo/annotations/classes.json',
#     'video_folder': '../../../datasets/slovo/video', 
#     'sample_size': 224, 
#     'sample_stride': 2, 
#     'sample_n_frames': 16,
# }
# dataset = ValidVideoDataset(**params)

# item = dataset[4]
# images = item['pixel_values']
# images = (images * 0.5 + 0.5).permute(0, 2, 3, 1).clamp(0, 1)
# images = images.float().numpy()
# images = np.clip(images * 255, 0, 255).astype(np.uint8)

# show_images(images[:4], figsize=(15, 15), titles=[item['label']] * 4)