import timm
import torch.nn as nn

class TimmNet(nn.Module):
    def __init__(self, model_name, num_classes, in_chans=3, pretrained=False, global_pool='max'):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, 
            num_classes=num_classes, 
            in_chans=in_chans, 
            pretrained=pretrained,
            global_pool=global_pool,
        )
    def forward(self, x):
        return self.model(x)
