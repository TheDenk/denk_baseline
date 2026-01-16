import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from einops import rearrange
from functools import partial
from torch import nn, einsum

from segmentation_models_pytorch.unet.decoder import UnetDecoder


class CoatNet(nn.Module):
    def __init__(self, encoder_dim=[152, 320, 320, 320, 320]):
        super().__init__()
        self.enc = CoaT(
            patch_size=4,
            embed_dims=encoder_dim,
            serial_depths=[2, 2, 2, 2, 2],
            parallel_depth=6,
            num_heads=8,
            mlp_ratios=[4, 4, 4, 4, 4],
            out_features=[f'x{i+1}_nocls' for i in range(5)],
        )
        
        decoder_dim = [256, 128, 64, 32, 16]
        conv_dim = 32

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.dec = UnetDecoder(
                    encoder_channels=[0, conv_dim] + encoder_dim,
                    decoder_channels=decoder_dim,
                    n_blocks=5,
                    use_batchnorm=True,
                    center=False,
                    attention_type=None,
                )

        self.logit = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1),
            )
        self.aux = nn.ModuleList([
            nn.Conv2d(e_dim, 1, kernel_size=1, padding=0) for e_dim in encoder_dim
        ])
    
    def forward(self, batch):
        x = batch.get('inputs', batch.get('image'))
        b, c, h, w = x.shape
        enc_x = self.enc(x)
        conv = self.conv(x)
        
        enc_feats = [enc_x[f'x{i+1}_nocls'] for i in range(5)]
        
        feature = enc_feats[::-1] 
        head = feature[0]
        skip = feature[1:] + [conv, None]
        d = self.dec.center(head)

        dec_x = []
        for i, decoder_block in enumerate(self.dec.blocks):
            s = skip[i]
            d = decoder_block(d, s)
            dec_x.append(d)
            
        x = dec_x[-1]
        
        x = self.logit(x)
        
        device = x.device
        x = x.to(torch.float32)
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.to(device)
        
        return x


class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            window = {window: h}                                                         # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()            
        
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),                          
                groups=cur_head_split*Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == 1 + H * W

        # Convolutional relative position encoding.
        q_img = q[:,:,1:,:]                                                              # Shape: [B, h, H*W, Ch].
        v_img = v[:,:,1:,:]                                                              # Shape: [B, h, H*W, Ch].
        
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)                      # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q_img * conv_v_img
        zero = torch.zeros((B, h, 1, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
        EV_hat = torch.cat((zero, EV_hat_img), dim=2)                                # Shape: [B, h, N, Ch].

        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)                                           # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)                                           # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x                                                                         # Shape: [B, N, C].


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding. 
        Note: This module is similar to the conditional position encoding in CPVT.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim) 
    
    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W

        # Extract CLS token and image tokens.
        cls_token, img_tokens = x[:, :1], x[:, 1:]                                       # Shape: [B, 1, C], [B, H*W, C].
        
        # Depthwise convolution.
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        # Combine with CLS token.
        x = torch.cat((cls_token, x), dim=1)

        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 shared_cpe=None, shared_crpe=None):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size):
        # Conv-Attention.
        x = self.cpe(x, size)                  # Apply convolutional position encoding.
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)   # Apply factorized attention and convolutional relative position encoding.
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class ParallelBlock(nn.Module):
    """ Parallel block class. """
    def __init__(self, dims, num_heads, mlp_ratios=[], qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 shared_cpes=None, shared_crpes=None):
        super().__init__()

        # Conv-Attention.
        self.cpes = shared_cpes

        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.norm15 = norm_layer(dims[4])
        
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(
            dims[1], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpes[1]
        )
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(
            dims[2], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpes[2]
        )
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(
            dims[3], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpes[3]
        )
        self.factoratt_crpe5 = FactorAtt_ConvRelPosEnc(
            dims[4], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpes[4]
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])
        self.norm25 = norm_layer(dims[4])
        
        assert dims[1] == dims[2] == dims[3] == dims[4]                              # In parallel block, we assume dimensions are the same and share the linear transformation.
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3] == mlp_ratios[4]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = self.mlp5 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def upsample(self, x, output_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W

        cls_token  = x[:, :1, :]
        img_tokens = x[:, 1:, :]
        
        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        device = img_tokens.device
        img_tokens = img_tokens.to(torch.float32)
        img_tokens = F.interpolate(img_tokens, size=output_size, mode='bilinear')  # FIXME: May have alignment issue.
        img_tokens =img_tokens.to(device)
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)
        
        out = torch.cat((cls_token, img_tokens), dim=1)

        return out

    def forward(self, x1, x2, x3, x4, x5, sizes):
        _, (H2, W2), (H3, W3), (H4, W4), (H5, W5)  = sizes

        # Conv-Attention.
        x2 = self.cpes[1](x2, size=(H2, W2))  # Note: x1 is ignored.
        x3 = self.cpes[2](x3, size=(H3, W3))
        x4 = self.cpes[3](x4, size=(H4, W4))
        x5 = self.cpes[4](x5, size=(H5, W5))

        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur5 = self.norm15(x5)

        cur2 = self.factoratt_crpe2(cur2, size=(H2,W2))
        cur3 = self.factoratt_crpe3(cur3, size=(H3,W3))
        cur4 = self.factoratt_crpe4(cur4, size=(H4,W4))
        cur5 = self.factoratt_crpe5(cur5, size=(H5,W5))

        upsample3_2 = self.upsample(cur3, output_size=(H2,W2), size=(H3,W3))
        upsample4_3 = self.upsample(cur4, output_size=(H3,W3), size=(H4,W4))
        upsample4_2 = self.upsample(cur4, output_size=(H2,W2), size=(H4,W4))
        upsample5_4 = self.upsample(cur5, output_size=(H4,W4), size=(H5,W5))
        upsample5_3 = self.upsample(cur5, output_size=(H3,W3), size=(H5,W5))
        upsample5_2 = self.upsample(cur5, output_size=(H2,W2), size=(H5,W5))

        downsample2_3 = self.downsample(cur2, output_size=(H3,W3), size=(H2,W2))
        downsample3_4 = self.downsample(cur3, output_size=(H4,W4), size=(H3,W3))
        downsample2_4 = self.downsample(cur2, output_size=(H4,W4), size=(H2,W2))
        downsample4_5 = self.downsample(cur4, output_size=(H5,W5), size=(H4,W4))
        downsample3_5 = self.downsample(cur3, output_size=(H5,H5), size=(H3,W3))
        downsample2_5 = self.downsample(cur2, output_size=(H5,H5), size=(H2,W2))


        cur2 = cur2  + upsample3_2   + upsample4_2
        cur3 = cur3  + upsample4_3   + downsample2_3
        cur4 = cur4  + upsample5_4   + downsample2_4
        cur5 = cur5  + downsample4_5 + downsample2_5
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        x5 = x5 + self.drop_path(cur5)

        # MLP.
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur5 = self.norm25(x5)
        
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        cur5 = self.mlp5(cur5)
        
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        x5 = x5 + self.drop_path(cur5)

        return x1, x2, x3, x4, x5


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]

        x = self.proj(x).flatten(2).transpose(1, 2)
        out = self.norm(x)
        
        return out, (out_H, out_W)


class CoaT(nn.Module):
    """ CoaT class. """
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[0, 0, 0, 0], 
                 serial_depths=[0, 0, 0, 0], parallel_depth=0,
                 num_heads=0, mlp_ratios=[0, 0, 0, 0], qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 return_interm_layers=False, out_features=None, crpe_window={3:2, 5:3, 7:3},
                 **kwargs):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes

        # Patch embeddings.
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.patch_embed5 = PatchEmbed(patch_size=2, in_chans=embed_dims[3], embed_dim=embed_dims[4])

        # Class tokens.
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.cls_token5 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # Convolutional position encodings.
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)
        self.cpe5 = ConvPosEnc(dim=embed_dims[4], k=3)

        # Convolutional relative position encodings.
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)
        self.crpe5 = ConvRelPosEnc(Ch=embed_dims[4] // num_heads, h=num_heads, window=crpe_window)

        # Enable stochastic depth.
        dpr = drop_path_rate
        
        # Serial blocks 1.
        self.serial_blocks1 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, 
                shared_cpe=self.cpe1, shared_crpe=self.crpe1
            )
            for _ in range(serial_depths[0])]
        )

        # Serial blocks 2.
        self.serial_blocks2 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, 
                shared_cpe=self.cpe2, shared_crpe=self.crpe2
            )
            for _ in range(serial_depths[1])]
        )

        # Serial blocks 3.
        self.serial_blocks3 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, 
                shared_cpe=self.cpe3, shared_crpe=self.crpe3
            )
            for _ in range(serial_depths[2])]
        )

        # Serial blocks 4.
        self.serial_blocks4 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, 
                shared_cpe=self.cpe4, shared_crpe=self.crpe4
            )
            for _ in range(serial_depths[3])]
        )
        
        self.serial_blocks5 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[4], num_heads=num_heads, mlp_ratio=mlp_ratios[4], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, 
                shared_cpe=self.cpe5, shared_crpe=self.crpe5
            )
            for _ in range(serial_depths[3])]
        )

        # Parallel blocks.
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([
                ParallelBlock(
                    dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, 
                    shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4, self.cpe5],
                    shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4, self.crpe5]
                )
                for _ in range(parallel_depth)]
            )

        # Initialize weights.
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)
        trunc_normal_(self.cls_token5, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4', 'cls_token5'}

    def insert_cls(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def remove_cls(self, x):
        """ Remove CLS token. """
        return x[:, 1:, :]

    def forward_features(self, x0):
        B = x0.shape[0]

        # Serial blocks 1.
        x1, (H1, W1) = self.patch_embed1(x0)
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_cls(x1)
        x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        # Serial blocks 2.
        x2, (H2, W2) = self.patch_embed2(x1_nocls)
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_cls(x2)
        x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 3.
        x3, (H3, W3) = self.patch_embed3(x2_nocls)
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_cls(x3)
        x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 4.
        x4, (H4, W4) = self.patch_embed4(x3_nocls)
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_cls(x4)
        x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        
        # Serial blocks 4.
        x5, (H5, W5) = self.patch_embed5(x4_nocls)
        x5 = self.insert_cls(x5, self.cls_token5)
        for blk in self.serial_blocks5:
            x5 = blk(x5, size=(H5, W5))
        x5_nocls = self.remove_cls(x5)
        x5_nocls = x5_nocls.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous()

        # Parallel blocks.
        for blk in self.parallel_blocks:
            x1, x2, x3, x4, x5 = blk(x1, x2, x3, x4, x5, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4), (H5, W5)])

        feat_out = {}   
        if 'x1_nocls' in self.out_features:
            x1_nocls = self.remove_cls(x1)
            x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            feat_out['x1_nocls'] = x1_nocls
        if 'x2_nocls' in self.out_features:
            x2_nocls = self.remove_cls(x2)
            x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
            feat_out['x2_nocls'] = x2_nocls
        if 'x3_nocls' in self.out_features:
            x3_nocls = self.remove_cls(x3)
            x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
            feat_out['x3_nocls'] = x3_nocls
        if 'x4_nocls' in self.out_features:
            x4_nocls = self.remove_cls(x4)
            x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
            feat_out['x4_nocls'] = x4_nocls
        if 'x5_nocls' in self.out_features:
            x5_nocls = self.remove_cls(x5)
            x5_nocls = x5_nocls.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous()
            feat_out['x5_nocls'] = x5_nocls
        return feat_out


    def forward(self, x):
        return self.forward_features(x)


class CoatNet(nn.Module):
    def __init__(self, in_channels, num_classes, encoder_dim=[152, 320, 320, 320, 320]):
        super().__init__()
        self.enc = CoaT(
            in_chans=in_channels,
            patch_size=4,
            embed_dims=encoder_dim,
            serial_depths=[2, 2, 2, 2, 2],
            parallel_depth=6,
            num_heads=8,
            mlp_ratios=[4, 4, 4, 4, 4],
            out_features=[f'x{i+1}_nocls' for i in range(5)],
        )
        
        decoder_dim = [256, 128, 64, 32, 16]
        conv_dim = 32

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.dec = UnetDecoder(
                    encoder_channels=[0, conv_dim] + encoder_dim,
                    decoder_channels=decoder_dim,
                    n_blocks=5,
                    use_batchnorm=True,
                    center=False,
                    attention_type=None,
                )

        self.logit = nn.Sequential(
                nn.Conv2d(32, num_classes, kernel_size=1),
            )
        # self.aux = nn.ModuleList([
        #     nn.Conv2d(e_dim, 1, kernel_size=1, padding=0) for e_dim in encoder_dim
        # ])
    
    def forward(self, x):
        b, c, h, w = x.shape
        enc_x = self.enc(x)
        conv = self.conv(x)
        
        enc_feats = [enc_x[f'x{i+1}_nocls'] for i in range(5)]
        
        feature = enc_feats[::-1] 
        head = feature[0]
        skip = feature[1:] + [conv, None]
        d = self.dec.center(head)

        dec_x = []
        for i, decoder_block in enumerate(self.dec.blocks):
            s = skip[i]
            d = decoder_block(d, s)
            dec_x.append(d)
            
        x = dec_x[-1]
        
        x = self.logit(x)
        
        device = x.device
        x = x.to(torch.float32)
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.to(device)
        return x
    

def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = {}
    for name, weights in ckpt['state_dict'].items():
        state_dict[name] = weights
    return state_dict


def get_coat(in_channels, classes, pretrained=None, **kwargs):
    model = CoatNet(in_channels, classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict(pretrained)
        model.load_state_dict(state_dict)
    return model