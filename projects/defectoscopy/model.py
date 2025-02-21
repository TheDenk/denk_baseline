import timm
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, model_name, in_chans=3, num_classes=2, pretrained=True):
        super(CustomModel, self).__init__()
        self.backbone = timm.create_model(model_name, in_chans=in_chans, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, batch):
        out = self.backbone(batch)
        return out
        