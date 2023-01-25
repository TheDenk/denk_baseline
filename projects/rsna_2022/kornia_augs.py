import torch
import torch.nn as nn
import numpy as np

from kornia.augmentation import AugmentationBase2D, RandomHorizontalFlip, RandomVerticalFlip, \
RandomAffine, RandomThinPlateSpline, RandomGamma, RandomContrast, RandomBrightness, ImageSequential, RandomCrop


def generate_parameter(min_p, max_p, name=None):
    if min_p == max_p:
        return min_p

    elif isinstance(min_p, float) and isinstance(max_p, float):
        return np.random.uniform(min_p, max_p)

    elif isinstance(min_p, int) and isinstance(max_p, int):
        return np.random.randint(min_p, max_p)

    else:
        raise Exception(f'Generate random parameter {name} error. Set both type of parameters int or float.')

def generate_rect_coordinates(img_h: int, img_w: int,
                              min_x: int = None, min_y: int = None,
                              max_x: int = None, max_y: int = None,
                              min_h: int = None, min_w: int = None,
                              max_h: int = None, max_w: int = None):
    min_h = 1 if min_h is None else min_h
    min_w = 1 if min_w is None else min_w

    max_h = img_h if max_h is None else max_h
    max_w = img_w if max_w is None else max_w

    rect_h = generate_parameter(min_h, max_h)
    rect_w = generate_parameter(min_w, max_w)

    min_x = 0 if min_x is None or min_x < 0 else min_x
    min_y = 0 if min_y is None or min_y < 0 else min_y

    max_x = img_w - rect_w if max_x is None else max_x
    max_y = img_h - rect_h if max_y is None else max_y

    x1 = generate_parameter(min_x, max_x)
    x2 = x1 + rect_w if x1 + rect_w <= img_w else img_w

    y1 = generate_parameter(min_y, max_y)
    y2 = y1 + rect_h if y1 + rect_h <= img_h else img_h
    return x1, y1, x2, y2

    
class SameCutout(AugmentationBase2D):
    def __init__(self, 
                 num_holes: list = [4, 8], 
                 mm_size:list = [0.05, 0.15], 
                 same_on_batch: bool = False,
                 p: float = 1.0,
                 keepdim: bool = False,
                ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.num_holes = num_holes
        self.mm_size = mm_size
        
    def get_masks(self, batch_shape):
        bs, c, h, w = batch_shape
        
        min_s, max_s = self.mm_size
        min_h, max_h = int(h*min_s), int(h*max_s)
        min_w, max_w = int(w*min_s), int(w*max_s)
        masks = torch.ones(*batch_shape)
        
        nh_min, nh_max = self.num_holes
        num_holes = generate_parameter(nh_min, nh_max)
        
        for _ in range(num_holes):
            rect = generate_rect_coordinates(
                img_h=h,
                img_w=w,
                min_h=min_h, min_w=min_w,
                max_h=max_h, max_w=max_w,
            )
            x1, y1, x2, y2 = rect
            masks[:, :, y1:y2, x1:x2] = 0        

        return masks
    
    def compute_transformation(self, x, params, flags):
        return torch.tensor(1.0)

    def apply_transform(self, images, params, flags, transform):
        masks = self.get_masks(images.shape).type_as(images)
        images = images * masks
        return images

    
class SameRoll(AugmentationBase2D):
    def __init__(self, 
                 mm_roll:list = [0.1, 0.9], 
                 same_on_batch: bool = False,
                 p: float = 1.0,
                 keepdim: bool = False,
                ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.mm_roll = mm_roll
    
    def compute_transformation(self, x, params, flags):
        return torch.tensor(1.0)

    def apply_transform(self, images, params, flags, transform):
        bs, c, h, w = images.shape
        min_r, max_r = self.mm_roll
        shift_y = torch.randint(int(h*min_r), int(h*max_r), (1,))
        shift_x = torch.randint(int(w*min_r), int(w*max_r), (1,))
        
        images = torch.roll(images, shifts=(shift_y, shift_x), dims=(2, 3))
        return images

    
class KorniaAugs(nn.Module):
    def __init__(self, img_h, img_w):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        
        self.flip = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        )
        
        p=0.99
        self.crop = ImageSequential(
            RandomCrop((img_h - img_h // 32, img_w - img_w // 32), p=p),
            RandomCrop((img_h - img_h // 24, img_w - img_w // 24), p=p),
            RandomCrop((img_h - img_h // 20, img_w - img_w // 20), p=p),
            RandomCrop((img_h - img_h // 16, img_w - img_w // 16), p=p),
            RandomCrop((img_h - img_h // 12, img_w - img_w // 12), p=p),
            random_apply=1,
        )

        p=0.99
        self.transform_geometry = ImageSequential(
            RandomAffine(degrees=20, translate=0.05, scale=[0.95,1.05], shear=5, p=p),
            RandomThinPlateSpline(scale=0.05, p=p),
            random_apply=1,
        )

        p=0.99
        self.transform_intensity = ImageSequential(
            # RandomGamma(gamma=(0.5, 1.5), gain=(0.5, 1.2), p=p),
            RandomContrast(contrast=(0.9,1.1), p=p),
            RandomBrightness(brightness=(0.9,1.1), p=p),
            random_apply=1,
        )

        p=0.99
        self.transform_custom = ImageSequential(
            SameRoll(p=p), 
            SameCutout(p=p),
            random_apply=1,
        )

    @torch.no_grad() 
    def forward(self, x):
        x = self.flip(x)
        x = self.crop(x)
        x = self.transform_geometry(x)
        x = self.transform_intensity(x)
        x = self.transform_custom(x)
        return x