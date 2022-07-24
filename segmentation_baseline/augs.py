import cv2
import albumentations as A


def get_train_augs(patch_h=None, patch_w=None):
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            A.RandomCrop(512, 512, p=1.0), 
            A.RandomCrop(1024, 1024, p=1.0),
            A.RandomCrop(1536, 1536, p=1.0),
            A.RandomCrop(2048, 2048, p=1.0),
        ], p=0.75),
        A.ToGray(p=0.15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.GaussianBlur(p=0.5),
        ], p=0.25),
    ], p=1.0)


def get_valid_augs(patch_h=None, patch_w=None):
    # return A.Compose([
    #         A.Resize(1024, 1024, interpolation=cv2.INTER_NEAREST, p=1),
    #     ], p=1.0)
    return None

def get_test_augs(patch_h=None, patch_w=None):
    # return A.Compose([
    #         A.Resize(1024, 1024, interpolation=cv2.INTER_NEAREST, p=1),
    #     ], p=1.0)
    return None