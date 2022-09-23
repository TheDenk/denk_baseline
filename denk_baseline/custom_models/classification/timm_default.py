import timm
import torch
import torch.nn as nn

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
    def forward(self, x):
        x = self.backbone(x)
        # x = torch.sigmoid(x)
        # x = (x > 0.5).float()
        return x
