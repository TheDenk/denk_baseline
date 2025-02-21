import os
import random
import glob

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data.dataset import Dataset

from .utils import construct


class DefectoscopyClassificationDataset(Dataset):
    def __init__(
        self,
        defects_csv_dir,
        nondefects_csv_dir,
        window_size=512,
        min_df_size=256,
        is_train=False,
        dilate_kernel=None,
    ):
        defects_csv_paths = glob.glob(os.path.join(defects_csv_dir, "*.csv"))
        nondefects_csv_paths = glob.glob(os.path.join(nondefects_csv_dir, "*.csv"))
        defects_csv_paths = defects_csv_paths[:-50] if is_train else defects_csv_paths[-50:]
        nondefects_csv_paths = nondefects_csv_paths[:-50] if is_train else nondefects_csv_paths[-50:]

        self.dataframes = self.read_csv_files(defects_csv_paths, 1, min_df_size) + self.read_csv_files(nondefects_csv_paths, 0, min_df_size)
        random.shuffle(self.dataframes)
        
        self.window_size = window_size
        self.transforms = get_train_transform() if is_train else get_valid_transform()
        self.dilate_kernel = dilate_kernel
        self.length = len(self.dataframes)

    def read_csv_files(self, csv_paths, label, min_df_size):
        dataframes = [
            pd.read_csv(x, usecols=["coord_system", "coord_display", "channel", "signal", "delay", "amplitude"]) for x in csv_paths
        ]
        dataframes = [(x, label) for x in dataframes if x.shape[0] > min_df_size]
        return dataframes
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        df, label = self.dataframes[idx]
        sample = construct(df)
        sample = np.concatenate([np.expand_dims(sample[:, :642], -1), np.expand_dims(sample[:, 642:], -1)], axis=2)
        if self.dilate_kernel:
            kernel = np.ones(self.dilate_kernel, dtype=np.float32)
            sample = cv2.filter2D(sample.astype(np.float32), -1, kernel)
        sample = torch.from_numpy(sample).permute(2, 0, 1) / 15.0
        if self.transforms:
            sample = self.transforms(sample)
        
        return {"image": sample, "label": label}


def get_train_transform():
    transforms = v2.Compose([
        # v2.RandomCrop(size=(200, 512), p=0.7),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(-5, 5)),
        v2.Resize([210, 642]),
    ])
    return transforms

def get_valid_transform():
    transforms = v2.Compose([
        v2.Resize([210, 642]),
    ])
    return transforms
