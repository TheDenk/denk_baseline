import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimmNet(nn.Module):
    def __init__(self, model_name, num_classes, in_chans=3, pretrained=False, global_pool='max'):
        super().__init__()
        self.backbone = timm.create_model(
            model_name=model_name, 
            num_classes=num_classes, 
            in_chans=in_chans, 
            pretrained=pretrained,
            global_pool=global_pool,
        )
        self.cancer = nn.Linear(1408,1)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)
        # print(x.shape)
        x = self.cancer(x)
        # x = x.reshape(-1)

        x = F.sigmoid(x)
        # x = torch.nan_to_num(x)

        return x
