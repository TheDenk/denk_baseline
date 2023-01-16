import cv2
import albumentations as A

from denk_baseline.augs import BaseAugs


class TrainAugs(BaseAugs):
    def get_augs(self):
        return A.Compose([
            A.Resize(1024, 1024, always_apply=True),
            A.OneOf([
                A.RandomCrop(768, 768, p=1.0),
                A.RandomCrop(768 + 64, 768 + 64, p=1.0),
                A.RandomCrop(768 + 128, 768 + 128, p=1.0),
                A.RandomCrop(768 + 192, 768 + 192, p=1.0),
            ], p=0.8),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),
            A.OneOf([
                # A.OpticalDistortion(p=1.0),
                # A.GridDistortion(p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.5),
            A.Resize(1024, 1024, always_apply=True),
        ], p=1.0)