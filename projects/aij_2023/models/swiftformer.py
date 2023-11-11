# https://github.com/Amshaker/SwiftFormer/blob/main/models/swiftformer.py
"""
SwiftFormer
"""
import os
import math
import copy
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.layers.helpers import to_2tuple
import einops


SwiftFormer_width = {
    'XS': [48, 56, 112, 220],
    'S': [48, 64, 168, 224],
    'l1': [48, 96, 192, 384],
    'l3': [64, 128, 320, 512],
}

SwiftFormer_depth = {
    'XS': [3, 3, 6, 4],
    'S': [3, 3, 9, 6],
    'l1': [4, 3, 10, 5],
    'l3': [4, 4, 12, 6],
}

def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=embed_dim, eps=1e-6, affine=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True, num_groups=32):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(num_features=dim, eps=1e-6, affine=True)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.0, num_groups=32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(num_features=in_features, eps=1e-6, affine=True)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out


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
        self.frame_count = frame_count
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
        bf, c, h, w = x.shape
        b = bf // self.frame_count
        x = x.unsqueeze(1).reshape(b, self.frame_count, c, h, w).contiguous()
        x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.frame_count, c).contiguous()

        x = self.norm_inner(self.proj_inner(x))

        for attantion_block in self.attantion_blocks:
            x = attantion_block(x) + x
        x = self.proj_out(self.norm_out(x))

        x = x.unsqueeze(1).unsqueeze(1).reshape(b, h, w, self.frame_count, c).permute(0, 4, 3, 1, 2).contiguous()
        x = x.reshape(b*self.frame_count, 1, c, h, w).squeeze(1).contiguous()
        x = x + residual
        return x
    

class Conv3DBlock(nn.Module):
    def __init__(self, channels, frame_count, pool_kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        self.frame_count = frame_count
        self.norm = nn.BatchNorm3d(num_features=channels, eps=1e-6, affine=True)
        self.conv = nn.Conv3d(
            channels,
            channels,
            pool_kernel,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )
    
    def forward(self, x):
        residual = x
        bf, c, h, w = x.shape
        b = bf // self.frame_count
        x = x.unsqueeze(1).reshape(b, self.frame_count, c, h, w).contiguous()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = self.norm(x)
        x = self.conv(x)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(b*self.frame_count, 1, c, h, w).squeeze(1).contiguous()
        x = x + residual
        return x


class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(num_features=dim, eps=1e-6, affine=True)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class SwiftFormerEncoder(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = SwiftFormerLocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(
                    0, 3, 1, 2))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))
        return x


def Stage(dim, index, layers, mlp_ratio=4.,
          act_layer=nn.GELU,
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, 
          vit_num=1, temporal_indices=None, conv3d_indices=None, frame_count=32):
    """
    Implementation of each SwiftFormer stages. Here, SwiftFormerEncoder used as the last block in all stages, while ConvEncoder used in the rest of the blocks.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        if layers[index] - block_idx <= vit_num:
            blocks.append(SwiftFormerEncoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))
        else:
            blocks.append(ConvEncoder(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))
    
    if index in temporal_indices:
        blocks.append(TemporalAttnetionBlock(frame_count=frame_count, in_channels=dim))

    if index in conv3d_indices:
        blocks.append(Conv3DBlock(frame_count=frame_count, channels=dim))

    blocks = nn.Sequential(*blocks)
    return blocks


class SwiftFormer(nn.Module):
    def __init__(self, arch='XS', temporal_indices=None,
                 mlp_ratios=4, downsamples=None, conv3d_indices=None,
                 act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 init_cfg=None,
                 vit_num=1,
                 frame_count=32,
                 **kwargs):
        super().__init__()
        layers = SwiftFormer_depth[arch]
        embed_dims = SwiftFormer_width[arch]

        self.frame_count = frame_count
        self.num_classes = num_classes
        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          act_layer=act_layer,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          vit_num=vit_num, temporal_indices=temporal_indices, conv3d_indices=conv3d_indices,
                          frame_count=self.frame_count)
            network.append(stage)

            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(num_features=embed_dims[-1], eps=1e-6, affine=True)
        
        self.out_conv = Conv3DBlock(frame_count=frame_count, channels=embed_dims[-1])
        self.out_norm = nn.BatchNormrd(num_features=embed_dims[-1], eps=1e-6, affine=True)
        
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()

        # self.apply(self.cls_init_weights)
        self.apply(self._init_weights)

        # self.init_cfg = copy.deepcopy(init_cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).contiguous()
        
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        x = self.norm(x)  
        x = self.out_conv(x)
        x = self.out_norm(x)
        _, c_, h_, w_ = x.shape
        x = x.unsqueeze(1).reshape(b, c_, f, h_, w_).contiguous()
        
        x = x.flatten(2).mean(-1)
        cls_out = self.head(x)
        
        return cls_out

if __name__ == '__main__':
    device = 'cpu'
    torch_dtype = torch.float32
    batch_shape = [1, 3, 32, 224, 224]

    model_kwargs = {
        'num_classes': 1001,
        'frame_count': 32,
        'arch': 'S',  # XS, S, l1, l3
        'temporal_indices': [1, 2, 3, 4, 5, 6],
        'conv3d_indices': [1, 2, 3, 4, 5, 6],
        'downsamples': [True, True, True, True],
        'vit_num': 1,
    }
    model = SwiftFormer(**model_kwargs).eval().to(device=device)
    batch = torch.rand(*batch_shape).to(device=device, dtype=torch_dtype)
    _ = model(batch)
    print('MODEL HAS BEEN CHECKED')