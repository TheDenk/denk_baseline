import cv2
import albumentations as A

from denk_baseline.augs import BaseAugs


class TrainAugs(BaseAugs):
    def get_augs(self):
        img_h, img_w = 2200, 1550
        return A.Compose([
            A.Resize(img_h, img_w, p=1.0),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.RandomCrop(512, 512, p=1.0),
                A.RandomCrop(768, 768, p=1.0),
                A.RandomCrop(1024, 1024, p=1.0),
                A.RandomCrop(1536, 1536, p=1.0),
            ], p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.35),
            A.OneOf([
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5),
                A.GaussianBlur(p=0.5),
            ], p=0.35),
        ], p=1.0)