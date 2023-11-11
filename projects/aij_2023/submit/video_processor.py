import torch
import numpy as np
import torchvision.transforms as transforms
from decord import VideoReader
import albumentations as A


def get_transforms(image):
    transform = A.Compose([
        A.LongestMaxSize(max_size=360, interpolation=1),
        A.PadIfNeeded(min_height=300, min_width=300, border_mode=0, value=(0,0,0)),
        A.CenterCrop(224, 224, always_apply=True, p=1.0)
    ], p=1.0)
    return transform(image=image)['image']


class VideoProcessor:
    def __init__(self, 
                 video_path, 
                 min_side=256,
                 sample_size=224, 
                 sample_stride=2, 
                 sample_n_frames=16,
                 batch_stride=16,
            ):
        
        self.video_reader = VideoReader(video_path)
        self.video_length = len(self.video_reader)

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.batch_stride = batch_stride
        
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
    
    def get_frames(self, start_frame, return_full, repeat_last):
        max_len = (self.sample_n_frames - 1) * self.sample_stride + 1
        clip_length = min(self.video_length - start_frame, max_len)
        if return_full:
            clip_length = self.video_length
        batch_index = np.linspace(start_frame, start_frame + clip_length - 1, self.sample_n_frames, dtype=int)
        if not return_full and repeat_last and clip_length < max_len:
            base_frames = np.arange(start_frame, clip_length, self.sample_stride)
            last_frame_count = self.sample_n_frames - len(base_frames)
            last_frame_index = clip_length - 1
            repeated_frame = np.ones(last_frame_count) * last_frame_index
            batch_index = np.concatenate([base_frames, repeated_frame])
        pixel_values = self.video_reader.get_batch(batch_index).asnumpy()
        pixel_values = np.stack([self.a_transforms(image=x)['image'] for x in pixel_values])
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        return pixel_values
    

    def get_video(self, start_frame=0, return_full=False, repeat_last=False):
        pixel_values = self.get_frames(start_frame, return_full, repeat_last)
        pixel_values = pixel_values / 255.0
        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values = pixel_values.permute(1, 0, 2, 3)
        return pixel_values
    
    def get_video_batch(self, return_full=False, repeat_last=False):
        video_batch = []
        clip_len = (self.sample_n_frames - 1) * self.sample_stride + 1
        for start_frame in range(0, max(self.video_length - clip_len, 1), self.batch_stride):
            frames = self.get_video(start_frame=start_frame, return_full=return_full, repeat_last=repeat_last)
            video_batch.append(frames)
        return video_batch
    