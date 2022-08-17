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
        # in_features = self.model.get_classifier().in_features
        # self.model.fc = nn.Sequential(
        #     nn.BatchNorm2d(in_features),
        #     nn.Linear(in_features=in_features, out_features=fc_dim),
        #     nn.GELU(),
        #     nn.BatchNorm2d(fc_dim),
        #     nn.Linear(in_features=fc_dim, out_features=num_classes),
        # )
    def forward(self, x):
        return self.model(x)
