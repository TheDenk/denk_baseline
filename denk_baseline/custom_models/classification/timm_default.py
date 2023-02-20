import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimmNet(nn.Module):
    def __init__(self, model_name, num_classes, in_chans=3, pretrained=False, global_pool='max', drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name=model_name, 
            num_classes=num_classes, 
            in_chans=in_chans, 
            pretrained=pretrained,
            global_pool=global_pool,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


def load_state_dict(weights_info,  map_location='cpu'):
    ckpt = torch.load(weights_info['ckpt_path'], map_location=map_location)

    ignore_layers = weights_info.get('ignore_layers', []) 
    ignore_layers = [
        'kornia_augs.transform_intensity.RandomContrast_0._param_generator.contrast_factor',
        'kornia_augs.transform_intensity.RandomBrightness_1._param_generator.brightness_factor',
    ]
    state_dict = {}
    for name, weights in ckpt['state_dict'].items():
        if name in ignore_layers:
            continue
        state_dict[name] = weights
    return state_dict


def get_timm(model_name, num_classes, weights_info=None, **kwargs):
    model = TimmNet(model_name, num_classes, **kwargs)
    if weights_info:
        state_dict = load_state_dict(weights_info)
        missing, unexpected = model.load_state_dict(state_dict)
        if missing:
            print('-'*10 + 'MISSING' + '-'*10)
            print(missing)
        if unexpected:
            print('-'*10 + 'UNEXPECTED' + '-'*10)
            print(unexpected)

    return model