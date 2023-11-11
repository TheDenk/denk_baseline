import torch


def crop_lt(x, crop_h, crop_w):
    return x[:, :, :, 0:crop_h, 0:crop_w]


def crop_lb(x, crop_h, crop_w):
    return x[:, :, :, -crop_h:, 0:crop_w]


def crop_rt(x, crop_h, crop_w):
    return x[:, :, :, 0:crop_h, -crop_w:]


def crop_rb(x, crop_h, crop_w):
    return x[:, :, :, -crop_h:, -crop_w:]


def center_crop(x, crop_h, crop_w):
    center_h = x.shape[3] // 2
    center_w = x.shape[4] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, :, y_min:y_max, x_min:x_max]


def flip_x(x):
    return x.flip(dims=(4,))

def one_flip(x):
    return torch.cat([x, flip_x(x)])

def five_crops(x, crop_h, crop_w):
    return torch.cat([f(x, crop_h, crop_w) for f in [crop_lt, crop_lb, crop_rt, crop_rb, center_crop]])


def ten_crops(x, crop_h, crop_w):
    five_crop = five_crops(x, crop_h, crop_w)
    ten_crops = torch.cat([five_crop, flip_x(five_crop)])
    return ten_crops