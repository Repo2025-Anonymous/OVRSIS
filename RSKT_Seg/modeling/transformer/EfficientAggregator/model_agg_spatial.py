import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Please install mamba_ssm to use Mamba layers.")


class VSSBlock(nn.Module):
    """Visual State Space Block (四向交叉扫描)"""
    def __init__(self, dim, d_state=16, expand=2, drop_path=0.):
        super().__init__()
        
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(d_model=dim, d_state=d_state, expand=expand)

        self.fuse_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), 
            nn.BatchNorm2d(dim), 
            nn.ReLU(inplace=True), 
            nn.Sigmoid()
            )

        self.reduce_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, guidance):
        B, C, H, W = x.shape

        x_fused = torch.cat([x, guidance], dim=1) 
        x_fused = self.reduce_conv(x_fused)
        
        xs_1 = rearrange(x_fused, 'b c h w -> b (h w) c')
        xs_2 = torch.flip(xs_1, dims=[1])
        xs_3 = rearrange(x_fused.transpose(2, 3), 'b c w h -> b (w h) c')
        xs_4 = torch.flip(xs_3, dims=[1])
        
        xs_batch = torch.stack([xs_1, xs_2, xs_3, xs_4], dim=0)
        xs_batch = rearrange(xs_batch, 'v b l c -> (v b) l c')

        xs_batch_out = self.mamba(self.norm(xs_batch))

        xs_out = rearrange(xs_batch_out, '(v b) l c -> v b l c', v=4)
        out_1 = xs_out[0]
        out_2 = xs_out[1]
        out_3 = xs_out[2]
        out_4 = xs_out[3]
        
        out_2 = torch.flip(out_2, dims=[1]) 
        out_4 = torch.flip(out_4, dims=[1]) 

        out_1 = rearrange(out_1, 'b (h w) c -> b c h w', h=H, w=W)
        out_2 = rearrange(out_2, 'b (h w) c -> b c h w', h=H, w=W)
        out_3 = rearrange(out_3, 'b (w h) c -> b c w h', w=W, h=H).transpose(2, 3)
        out_4 = rearrange(out_4, 'b (w h) c -> b c w h', w=W, h=H).transpose(2, 3)

        x_ssm = out_1 + out_2 + out_3 + out_4

        x_out = x_ssm + x_ssm * self.fuse_proj(x_ssm)
        x_out = x + self.drop_path(x_out)
        
        return x_out


class SpatialFusionBlock(nn.Module):
    def __init__(self, inter_channels, drop_path=0., **kwargs):
        super().__init__()
        self.sa = VSSBlock(dim=inter_channels, drop_path=drop_path)
    
    def forward(self, x, img_guidance):
        # x, img_guidance: [B, C, H, W]
        return self.sa(x, img_guidance)