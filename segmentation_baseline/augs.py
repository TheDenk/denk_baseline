import cv2
import albumentations as A


class BaseAugs:
    def __init__(self):
        self.augs = self.get_augs()
    def get_augs(self):
        return None
    def __call__(self, *args, **kwargs):
        return self.augs(*args, **kwargs)


class HubmapTrainAugs(BaseAugs):
    def get_augs(self):
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.RandomCrop(256, 256, p=1.0), 
                A.RandomCrop(512, 512, p=1.0), 
                A.RandomCrop(1024, 1024, p=1.0),
                A.RandomCrop(1536, 1536, p=1.0),
            ], p=1.0),
            A.ToGray(p=0.15),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5),
                A.GaussianBlur(p=0.5),
            ], p=0.25),
        ], p=1.0)


class NiiasTrainAugs(BaseAugs):
    def get_augs(self):
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.RandomCrop(256, 256, p=1.0), 
                A.RandomCrop(512, 512, p=1.0), 
            ], p=1.0),
            A.ToGray(p=0.15),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5),
                A.GaussianBlur(p=0.5),
            ], p=0.25),
        ], p=1.0)

