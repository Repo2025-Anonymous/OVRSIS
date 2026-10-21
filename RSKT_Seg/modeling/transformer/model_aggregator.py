import torch
import math
import random
import torch.nn as nn

from einops import rearrange, repeat
from .model_agg_spatial import SptialFusionBlock
from .model_agg_class import ClassFusionBlock

  
# class EffAggregatorLayer(nn.Module):
#     def __init__(self, hidden_dim, num_heads=8, drop_path=0., sr_ratio=2):
#         super().__init__()
#         self.spatial_fusion1 = SptialFusionBlock(hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)
#         self.spatial_fusion2 = SptialFusionBlock(hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)
#         self.class_fusion = ClassFusionBlock(hidden_dim, num_heads, drop_path=drop_path)

#     def forward(self, x, clip_img_guidance, dino_img_guidance, text_guidance):
#         B, _, T, H, W = x.shape
        
#         x = rearrange(x, 'B C T H W -> (B T) C H W')
#         clip_img_guidance = repeat(clip_img_guidance, "B C H W -> (B T) C H W", T=T)
#         dino_img_guidance = repeat(dino_img_guidance, "B C H W -> (B T) C H W", T=T)

#         # spatial fusion
#         x1 = self.spatial_fusion1(x, clip_img_guidance)
#         x2 = self.spatial_fusion2(x, dino_img_guidance)
#         x = x1 + x2

#         # class fusion
#         x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
#         x = self.class_fusion(x, text_guidance)
#         x = rearrange(x, '(B T) C H W -> B C T H W', B=B)

#         return x
    

class EffAggregatorLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, drop_path=0., sr_ratio=2):
        super().__init__()
        self.spatial_fusion1 = SptialFusionBlock(hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)
        self.spatial_fusion2 = SptialFusionBlock(hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)
        
        self.class_fusion1 = ClassFusionBlock(hidden_dim, num_heads, drop_path=drop_path)
        self.class_fusion2 = ClassFusionBlock(hidden_dim, num_heads, drop_path=drop_path)

    def forward(self, x, clip_img_guidance, dino_img_guidance, text_guidance):
        B, _, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        clip_img_guidance = repeat(clip_img_guidance, "B C H W -> (B T) C H W", T=T)
        dino_img_guidance = repeat(dino_img_guidance, "B C H W -> (B T) C H W", T=T)

        # spatial fusion
        x1 = self.spatial_fusion1(x, clip_img_guidance)
        x2 = self.spatial_fusion2(x, dino_img_guidance)

        # class fusion
        x1 = rearrange(x1, '(B T) C H W -> B C T H W', B=B)
        x1 = self.class_fusion1(x1, text_guidance)

        x2 = rearrange(x2, '(B T) C H W -> B C T H W', B=B)
        x2 = self.class_fusion2(x2, text_guidance)

        return x1 + x2