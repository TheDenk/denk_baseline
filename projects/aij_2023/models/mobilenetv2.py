'''MobilenetV2 in PyTorch.

See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
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

class PositionalEncoding(nn.Module):
    def __init__(self, in_dims, dropout = 0.0, max_len = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_dims, 2) * (-math.log(10000.0) / in_dims))
        pe = torch.zeros(1, max_len, in_dims)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalAttnetionLayer(nn.Module):
    def __init__(self, frame_count, in_channels=64, token_dim=512, dropout=0.1):
        super().__init__()
        self.frame_count = frame_count
        self.to_value = nn.Linear(in_channels, token_dim)
        self.to_query = nn.Linear(in_channels, token_dim)
        self.to_key = nn.Linear(in_channels, token_dim)

        self.scale_factor = token_dim ** -0.5
        self.to_out = nn.Linear(token_dim, in_channels)

        self.pos_encoder = PositionalEncoding(in_channels, dropout=dropout, max_len=frame_count)

    def forward(self, x):
        x = self.pos_encoder(x)
        residual = x

        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        query = torch.nn.functional.normalize(query, dim=-1) 
        key = torch.nn.functional.normalize(key, dim=-1) 
        value = torch.nn.functional.normalize(value, dim=-1) 

        mask = torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device)
        attention_scores = torch.baddbmm(
            mask,
            query,
            key.transpose(-1, -2),
            beta=0.0,
            alpha=1.0,
        ) * self.scale_factor
        del mask

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        x = torch.bmm(attention_probs, value)
        x = self.to_out(x)
        x = x + residual
        return x

class TemporalAttnetionBlock(nn.Module):
    def __init__(self, frame_count, in_channels=64, inner_dim=128, token_dim=512, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.proj_inner = nn.Linear(in_channels, inner_dim)
        self.norm_inner = nn.LayerNorm(inner_dim)

        self.attantion_blocks = nn.ModuleList(
            [
                TemporalAttnetionLayer(
                    frame_count, 
                    in_channels=inner_dim, 
                    token_dim=token_dim, 
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.norm_out = nn.LayerNorm(inner_dim)

    def forward(self, x):
        residual = x
        b, c, f, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, f, c).contiguous()

        x = self.norm_inner(self.proj_inner(x))

        for attantion_block in self.attantion_blocks:
            x = attantion_block(x) + x
        x = self.proj_out(self.norm_out(x))

        x = x.unsqueeze(1).unsqueeze(1).reshape(b, h, w, f, c).permute(0, 4, 3, 1, 2).contiguous()
        x = x + residual
        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, frame_count=32, sample_size=224, width_mult=1.0, temporal_indices=None, with_two_heads=True, ckpt_path=None, ignore_layers=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        last_channel = 1280
        self.with_two_heads = with_two_heads
        interverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1,1,1)],
            [6,  24, 2, (2,2,2)],
            [6,  32, 3, (2,2,2)],
            [6,  64, 4, (2,2,2)],
            [6,  96, 3, (1,1,1)],
            [6, 160, 3, (2,2,2)],
            [6, 320, 1, (1,1,1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        frame_count = int(frame_count * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, frame_count, (1,2,2))]
        # building inverted residual blocks
        for block_num, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1,1,1)
                self.features.append(block(frame_count, output_channel, stride, expand_ratio=t))
                frame_count = output_channel
            if block_num in temporal_indices:
                self.features.append(TemporalAttnetionBlock(frame_count=frame_count, in_channels=output_channel))
        # building last several layers
        self.features.append(conv_1x1x1_bn(frame_count, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        if with_two_heads:
            self.dist_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        if ckpt_path is not None:
            self.load_pretrain(ckpt_path, ignore_layers)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])

        if self.with_two_heads:
            cls_out = self.head(x.view(x.size(0), -1)), self.dist_head(x.view(x.size(0), -1))
        else:
            cls_out = self.head(x.view(x.size(0), -1))
        return cls_out

    def load_pretrain(self, ckpt_path, ignore_layers):
        ignore_layers = ignore_layers or []
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for layer_name, params in ckpt['state_dict'].items():
            if (layer_name in ignore_layers):
                continue
            state_dict[layer_name] = params
        m, u = self.load_state_dict(state_dict, strict=False)
        print(f'LOADED PRETRAIN MOBILENET 2 | MISSING KEYS: {len(m)} | UNEXPECTED KEYS: {len(u)}')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

    
def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNetV2(**kwargs)
    return model


