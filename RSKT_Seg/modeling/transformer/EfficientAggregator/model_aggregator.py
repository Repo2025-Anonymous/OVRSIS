import torch
import torch.nn as nn
from einops import rearrange, repeat
from .model_agg_spatial_ori import SptialFusionBlock
from .model_agg_class_ori import ClassFusionBlock


class FusionConv(nn.Module):
    """Fuse spatial and category-enhanced cost volumes.

    The module mirrors the parameter layout used by the released checkpoint:
    a 4-head cross attention followed by a 3x3 convolution over both streams.
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.fusion_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)

    def forward(self, spatial, category):
        # Both inputs are [B*T,C,H,W]. Cross-attention is performed per pixel.
        b, c, h, w = spatial.shape
        q = spatial.flatten(2).transpose(1, 2)
        kv = category.flatten(2).transpose(1, 2)
        attended, _ = self.cross_attn(q, kv, kv, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return self.fusion_conv(torch.cat([spatial + attended, category], dim=1))

  
class EffAggregatorLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, drop_path=0., sr_ratio=2):
        super().__init__()
        self.spatial_fusion = SptialFusionBlock(
            hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio
        )
        self.class_fusion = ClassFusionBlock(
            hidden_dim, num_heads, drop_path=drop_path
        )
        self.fusion_conv = FusionConv(hidden_dim, num_heads)

    def forward(self, x, img_guidance, text_guidance):
        B, _, T, H, W = x.shape
        
        # Spatial enhancement, followed by category enhancement.
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        img_guidance = repeat(img_guidance, "B C H W -> (B T) C H W", T=T)
        x = self.spatial_fusion(x, img_guidance)
        spatial = x
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        x = self.class_fusion(x, text_guidance)
        category = x
        x = self.fusion_conv(spatial, category)
        return rearrange(x, '(B T) C H W -> B C T H W', B=B)
