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
