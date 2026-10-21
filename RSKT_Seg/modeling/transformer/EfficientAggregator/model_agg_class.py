import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Please install mamba_ssm to use Mamba layers.")


class VSSBlock(nn.Module):
    """标准的 2D 视觉四向交叉扫描 Mamba 块"""
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [Batch, C, H, W]
        B, C, H, W = x.shape
        
        xs_1 = rearrange(x, 'b c h w -> b (h w) c')
        xs_2 = torch.flip(xs_1, dims=[1])
        xs_3 = rearrange(x.transpose(2, 3).contiguous(), 'b c w h -> b (w h) c')
        xs_4 = torch.flip(xs_3, dims=[1])
        
        xs_batch = torch.stack([xs_1, xs_2, xs_3, xs_4], dim=0)
        xs_batch = rearrange(xs_batch, 'v b l c -> (v b) l c')

        xs_batch_out = self.mamba(self.norm(xs_batch))

        xs_out = rearrange(xs_batch_out, '(v b) l c -> v b l c', v=4)
        out_1, out_2, out_3, out_4 = xs_out[0], xs_out[1], xs_out[2], xs_out[3]
        
        out_2 = torch.flip(out_2, dims=[1]) 
        out_4 = torch.flip(out_4, dims=[1]) 

        out_1 = rearrange(out_1, 'b (h w) c -> b c h w', h=H, w=W)
        out_2 = rearrange(out_2, 'b (h w) c -> b c h w', h=H, w=W)
        out_3 = rearrange(out_3, 'b (w h) c -> b c w h', h=H, w=W).transpose(2, 3).contiguous()
        out_4 = rearrange(out_4, 'b (w h) c -> b c w h', h=H, w=W).transpose(2, 3).contiguous()

        x_ssm = out_1 + out_2 + out_3 + out_4
        x_out = x_ssm + x_ssm * self.fuse_proj(x_ssm)
        x_out = x + self.drop_path(x_out)

        return x_out


# class OpenVocabMambaFusion(nn.Module):
#     """专门处理 5D 图像输入与 3D 文本嵌入的融合模块"""
#     def __init__(self, dim, drop_path=0., pooling_size=(2, 2)):
#         super().__init__()
#         self.dim = dim

#         self.q_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//4), nn.ReLU(), nn.Linear(dim//4, dim), nn.Sigmoid())
#         self.k_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//4), nn.ReLU(), nn.Linear(dim//4, dim), nn.Sigmoid())
#         self.v_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//4), nn.ReLU(), nn.Linear(dim//4, dim), nn.Sigmoid())

#         self.vss_block = VSSBlock(dim=dim, drop_path=drop_path)

#     def forward(self, img_feat, text_embeds):
#         # img_feat:    [B, C, N, H, W]
#         # text_embeds: [B, N, C]
#         B, C, N, H, W = img_feat.shape
        
#         img_feat = rearrange(img_feat, 'B C N H W -> (B N) C H W')
#         img_feat = rearrange(img_feat, '(B N) C H W -> (B H W) N C', N=N)
        
#         text_features = repeat(text_embeds, 'B N C -> (B H W) N C', H=H, W=W)

#         q = img_feat * self.q_proj(img_feat)             # [B, N*H*W, C] -> [5, 3456, 128]
#         k = text_features * self.k_proj(text_features)   # [B, N*H*W, C] -> [5, 3456, 128]
#         v = text_features * self.v_proj(text_features)   # [B, N*H*W, C] -> [5, 3456, 128]

#         attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, N*H*W, C]

#         x_fused = img_feat + attn_out
#         x_fused = rearrange(x_fused, '(B H W) N C -> (B N) C H W', N=N, H=H, W=W)
#         x_fused = self.vss_block(x_fused)
#         return x_fused


class OpenVocabMambaFusion(nn.Module):
    """专门处理 5D 图像输入与 3D 文本嵌入的融合模块"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dim = dim

        # GLU 风格的门控网络 (自带 Pre-Norm)
        self.q_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//4), nn.ReLU(inplace=True), nn.Linear(dim//4, dim), nn.Sigmoid())
        self.k_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//4), nn.ReLU(inplace=True), nn.Linear(dim//4, dim), nn.Sigmoid())
        self.v_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//4), nn.ReLU(inplace=True), nn.Linear(dim//4, dim), nn.Sigmoid())

        self.vss_block = VSSBlock(dim=dim, drop_path=drop_path)

    def forward(self, img_feat, text_embeds):
        # img_feat:    [B, C, N, H, W]
        # text_embeds: [B, N, C]
        B, C, N, H, W = img_feat.shape
        
        img_features = rearrange(img_feat, 'B C N H W -> (B H W) N C')
        text_features = repeat(text_embeds, 'B N C -> (B H W) N C', H=H, W=W)

        q = img_features * self.q_proj(img_features)            
        k = text_features * self.k_proj(text_features)  
        v = text_features * self.v_proj(text_features)  

        attn_out = F.scaled_dot_product_attention(q, k, v) 

        x_fused = img_features + attn_out
        x_fused = rearrange(x_fused, '(B H W) N C -> (B N) C H W', H=H, W=W)
        x_fused = self.vss_block(x_fused)
        
        return x_fused