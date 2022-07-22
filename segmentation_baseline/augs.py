import albumentations as A


def get_train_augs():
    return A.Compose([
            A.RandomCrop(512*1, 512*1, p=1),
            A.ToGray(p=0.15),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),
            A.Rotate(limit=180, border_mode=3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ], p=1.0)


def get_valid_augs():
    return None

def get_test_augs():
    return None