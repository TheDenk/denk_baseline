
import torch
import torch.nn as nn

from .head import MViTHead
from .mvit import MViT

# from pytorch_wavelets import DWTForward
    
# class EncoderDWT(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dwt = DWTForward(J=1, mode='zero', wave='db1')

#     def forward(self, x):
#         freq = self.img_to_dwt(x)
#         return freq

#     def img_to_dwt(self, img):
#         low, high = self.dwt(img)
#         b, _, _, h, w = high[0].size()
#         high = high[0].view(b, -1, h, w)
#         freq = torch.cat([low, high], dim=1)
#         return freq

class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MViTModel(torch.nn.Module):
    def __init__(self, num_classes=1001, backbone_channels=3, head_channels=768, arch='small', ckpt_path=None, ignore_layers=None, with_two_heads=False):
        super().__init__()
        self.backbone = MViT(arch=arch, in_channels=backbone_channels)
        self.cls_head = MViTHead(num_classes=num_classes, in_channels=head_channels)
        self.with_two_heads = with_two_heads
        if with_two_heads:
            self.cls_dist = MViTHead(num_classes=num_classes, in_channels=head_channels)
        
        if ckpt_path is not None:
            self.load_pretrain(ckpt_path, ignore_layers)

    def load_pretrain(self, ckpt_path, ignore_layers):
        ignore_layers = ignore_layers or []
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for layer_name, params in ckpt['state_dict'].items():
            if (layer_name in ignore_layers):
                continue
            state_dict[layer_name] = params
        m, u = self.load_state_dict(state_dict, strict=False)
        print(f'LOADED PRETRAIN MVIT | MISSING KEYS: {len(m)} | UNEXPECTED KEYS: {len(u)}')

    def forward(self, x):
        x = self.backbone(x)
        if self.with_two_heads:
            out = self.cls_head(x), self.cls_dist(x)
        else:
            out = self.cls_head(x)
        return out


MVIT_CUSTOM = {
    'small': {
        'num_classes': 1001,
        'backbone_channels': 3,
        'head_channels': 768,
        'arch': {
            'embed_dims': 96,
            'num_layers': 16,
            'num_heads': 1,
            'downscale_indices': [1, 3, 14]
        },
        'is_student': False,
        'ignore_layers': [],
        'ckpt_path': './pretrain_weights/custom-mvit-full-dwt-x1.ckpt',
    },
    'best_custom': {
        'num_classes': 1001,
        'backbone_channels': 64,
        'head_channels': 768,
        'arch': {
            'embed_dims': 96,
            'num_layers': 12,
            'num_heads': 1,
            'downscale_indices': [1, 3, 11]
        },
    }
}
BACKBONES = {
    'x0': torch.nn.Identity(),
    'x1': torch.nn.Sequential(
            InvertedResidual(inp=3, oup=16, stride=(1,1,1), expand_ratio=1),
            InvertedResidual(inp=16, oup=24, stride=(2,2,2), expand_ratio=2),
    ),
    'x2': torch.nn.Sequential(
            InvertedResidual(inp=3, oup=16, stride=(1,1,1), expand_ratio=1),
            InvertedResidual(inp=16, oup=24, stride=(2,2,2), expand_ratio=2),
            InvertedResidual(inp=24, oup=32, stride=(2,2,2), expand_ratio=4),
            InvertedResidual(inp=32, oup=64, stride=(1,1,1), expand_ratio=4),
    ),
    'x1_dwt': torch.nn.Sequential(
            InvertedResidual(inp=12, oup=12, stride=(1,1,1), expand_ratio=1),
            InvertedResidual(inp=12, oup=36, stride=(2,2,2), expand_ratio=2),
            InvertedResidual(inp=36, oup=72, stride=(1,1,1), expand_ratio=4),
    ),
}

HEADS = {
    'x0': torch.nn.Identity(),
}

class CustomMvitModel(torch.nn.Module):
    def __init__(self, mvit_kwargs, backbone, ckpt_path=None, ignore_layers=None, freeze_mvit=False):
        super().__init__()
        self.backbone = BACKBONES[backbone]
        self.mvit = MViTModel(**mvit_kwargs)

        if ckpt_path is not None:
            self.load_pretrain(ckpt_path, ignore_layers)

        if freeze_mvit:
            for param in self.mvit.parameters():
                param.requires_grad = False
            print('MVIT GRADS FREESED')
            
    def load_pretrain(self, ckpt_path, ignore_layers):
        ignore_layers = ignore_layers or []
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for layer_name, params in ckpt['state_dict'].items():
            if (layer_name in ignore_layers):
                continue
            state_dict[layer_name] = params
        m, u = self.load_state_dict(state_dict, strict=False)
        print(f'LOADED PRETRAIN CUSTOM MVIT | MISSING KEYS: {len(m)} | UNEXPECTED KEYS: {len(u)}')

    def freeze_mvit_blocks(self):
        for backbone_block in self.mvit.backbone.blocks:
            backbone_block.requires_grad_ = False

    def unfreeze_mvit_blocks(self):
        for backbone_block in self.mvit.backbone.blocks:
            backbone_block.requires_grad_ = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.mvit(x)
        return x
    