import torch
from torch import nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import os
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision import models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from utils import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from timm.models.registry import register_model

# No edits required
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# No edits required
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        
        super().__init__()
        
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample= None, #PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
#         x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
            print(x.shape)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        print("After forward features", x.shape)
#         x = self.head(x)
        return x


from ml_collections import config_dict
# # Resnet 18
# cfg = config_dict.ConfigDict()
# cfg.cnn_backbone = "resnet18"
# cfg.cnn_pyramid_fm  = [64,128,256,512]
# cfg.swin_pyramid_fm = [96, 192, 384, 768]
# cfg.image_size = 224
# cfg.patch_size = 4
# cfg.num_classes = 1000

# Resnet 50
# cfg = config_dict.ConfigDict()
# cfg.cnn_backbone = "resnet50"
# cfg.cnn_pyramid_fm  = [256,512,1024,2048]
# cfg.swin_pyramid_fm = [96, 192, 384, 768]
# cfg.image_size = 224
# cfg.patch_size = 4
# cfg.num_classes = 1000


class PyramidFeatures(nn.Module):
    def __init__(self, img_size = 224, in_channels = 3, cfg = None):
        super().__init__()
        self.swin_transformer = SwinTransformer(img_size,in_chans = 3)
        resnet = eval(f"models.{cfg.cnn_backbone}()")
        self.resnet_layers = nn.ModuleList(resnet.children())[:8]
        
        self.p1_ch = nn.Conv2d(cfg.cnn_pyramid_fm[0], cfg.swin_pyramid_fm[0] , kernel_size = 1)
        self.p1_pm = PatchMerging((cfg.image_size // cfg.patch_size, cfg.image_size // cfg.patch_size), cfg.swin_pyramid_fm[0])
        
        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(cfg.cnn_pyramid_fm[1], cfg.swin_pyramid_fm[1] , kernel_size = 1)
        self.p2_pm = PatchMerging((cfg.image_size // cfg.patch_size // 2, cfg.image_size // cfg.patch_size // 2), cfg.swin_pyramid_fm[1])
        
        
        self.proj1_2 = nn.Linear(cfg.swin_pyramid_fm[0], cfg.swin_pyramid_fm[1])
        self.proj3_4 = nn.Linear(cfg.swin_pyramid_fm[3], cfg.swin_pyramid_fm[2])
        
        
        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(cfg.cnn_pyramid_fm[2] , cfg.swin_pyramid_fm[2] , kernel_size =  1)
        self.p3_pm = PatchMerging((cfg.image_size // cfg.patch_size // 4,cfg.image_size // cfg.patch_size // 4), cfg.swin_pyramid_fm[2])
        
        
        self.p4 = self.resnet_layers[7]
        self.p4_ch = nn.Conv2d(cfg.cnn_pyramid_fm[3] , cfg.swin_pyramid_fm[3] , kernel_size = 1)
        

    def forward(self, x):
        
        for i in range(5):
            
            x = self.resnet_layers[i](x) 
        
        # 1
        fm1 = x
        fm1_ch = self.p1_ch(x)
        B, C, H, W = fm1_ch.shape
        fm1_reshaped = fm1_ch.view(B, C, W*H).permute(0,2,1)
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped  + sw1
        fm1_sw1 = self.p1_pm(sw1_skipped)
        
        #2
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        fm2 = self.p2(fm1)
        fm2_ch = self.p2_ch(fm2)
        B, C, H, W = fm2_ch.shape
        fm2_reshaped = fm2_ch.view(B, C, W*H).permute(0,2,1)
        fm2_sw2_skipped = fm2_reshaped  + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skipped)
    
        # Concat 1,2
        
        sw1_skipped_projected = self.proj1_2(sw1_skipped)
        concat1 = torch.cat((sw1_skipped_projected, fm2_sw2_skipped), dim = 1)
        
        
        #3
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        fm3 = self.p3(fm2)
        fm3_ch = self.p3_ch(fm3)
        B, C, H, W = fm3_ch.shape
        fm3_reshaped = fm3_ch.view(B, C, W*H).permute(0,2,1)
        fm3_sw3_skipped = fm3_reshaped  + fm2_sw3
        fm3_sw3 = self.p3_pm(fm3_sw3_skipped)
        
        #4
        fm3_sw4 = self.swin_transformer.layers[3](fm3_sw3)
        fm4 = self.p4(fm3)
        fm4_ch = self.p4_ch(fm4)
        B, C, H, W = fm4_ch.shape
        fm4_reshaped = fm4_ch.view(B, C, W*H).permute(0,2,1)
        fm4_sw4_skipped = fm4_reshaped  + fm3_sw4

        
        #concat 3,4
        sw4_skipped_projected = self.proj3_4(fm4_sw4_skipped)
        concat2 = torch.cat((sw4_skipped_projected, fm3_sw3_skipped), dim = 1)
        
        return [concat1, concat2], [sw1_skipped, fm2_sw2_skipped, fm3_sw3_skipped, fm4_sw4_skipped]

class All2Cross(nn.Module):
    def __init__(self, img_size = 224 , in_chans=3, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]), num_classes = 1000,
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, multi_conv=False, cfg = None):
        super().__init__()
        
        self.pyramid = PyramidFeatures(img_size= img_size, in_channels=in_chans, cfg = cfg)
        
        self.attention1 = Attention(embed_dim[0])
        self.attention2 = Attention(embed_dim[1])
        
        self.cls_token_1_2 = nn.Parameter(torch.zeros(1,1,embed_dim[0]))
        self.cls_token_3_4 = nn.Parameter(torch.zeros(1,1,embed_dim[1]))
        self.cls_token = nn.ParameterList([self.cls_token_1_2, self.cls_token_3_4])
        
        n_p1 = (cfg.image_size // cfg.patch_size) ** 2 + (cfg.image_size // cfg.patch_size // 2) ** 2 # default: 3920 
        n_p2 = (cfg.image_size // cfg.patch_size // 4) ** 2 + (cfg.image_size // cfg.patch_size // 8) ** 2  # default: 245 
        num_patches = (n_p1, n_p2)
        self.num_branches = 2
        
        
        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            trunc_normal_(self.cls_token[i], std=.02)
        self.apply(self._init_weights)
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        cls_token_1_2 = self.cls_token[0].expand(B, -1, -1) # stole cls_tokens impl from Phil Wang, thanks
        cls_token_3_4 = self.cls_token[1].expand(B, -1, -1) # stole cls_tokens impl from Phil Wang, thanks
        concats, skips = self.pyramid(x)
        concat1, concat2 = concats

        concat1 = torch.cat((cls_token_1_2, concat1), dim = 1)
        concat2 = torch.cat((cls_token_3_4, concat2), dim = 1)
        
        attn1 = self.attention1(concat1)
        attn2 = self.attention2(concat2)
        
        xs = [attn1, attn2]
        
        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]
        ce_logits = [self.head[i](x) for i, x in enumerate(out)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        return ce_logits


@register_model
def SwinRetinaResNet50(**kwargs):
    # Resnet 50
    cfg = config_dict.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024,2048]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    model = All2Cross(  img_size = 224 , in_chans=3, embed_dim=(192, 384),
                        depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]), num_classes = 1000,
                        num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        drop_path_rate=0., norm_layer=nn.LayerNorm, cfg = cfg) 
    model.default_cfg = _cfg()

    return model

@register_model
def SwinRetinaResNet18(**kwargs):
    # Resnet 18
    cfg = config_dict.ConfigDict()
    cfg.cnn_backbone = "resnet18"
    cfg.cnn_pyramid_fm  = [64,128,256,512]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4

    model = All2Cross(  img_size = 224 , in_chans=3, embed_dim=(192, 384),
                        depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]), num_classes = 1000,
                        num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        drop_path_rate=0., norm_layer=nn.LayerNorm, cfg = cfg) 
    model.default_cfg = _cfg()

    return model
