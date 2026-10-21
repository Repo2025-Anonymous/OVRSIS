# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import repeat


# class RSKT_Upsample(nn.Module):
#     """"Upscaling using feat from dino and clip"""
#     def __init__(self, 
#                 in_channels, 
#                 out_channels, 
#                 clip_guidance_channels, 
#                 dino_guidance_channels,
#                 use_remote_clip,
#                 use_remote_dino):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels - dino_guidance_channels, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_channels + clip_guidance_channels , out_channels)

#     def forward(self, x, clip_guidance, dino_guidance):
#         x = self.up(x)

#         T = x.size(0) // clip_guidance.size(0)
#         clip_guidance = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
#         x = torch.cat([x, clip_guidance], dim=1)

#         T = x.size(0) // dino_guidance.size(0)
#         dino_guidance = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)
#         x = torch.cat([x,dino_guidance], dim=1)
#         return self.conv(x)

# class DoubleConv(nn.Module):
#     """(convolution => [GN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(mid_channels // 16, mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(mid_channels // 16, mid_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


import torch
from torch import nn
from einops import repeat


class RSKT_Upsample(nn.Module):
    """Checkpoint-compatible guided upsampling block."""
    def __init__(self, in_channels, out_channels, clip_guidance_channels,
                 dino_guidance_channels, use_remote_clip=True,
                 use_remote_dino=True):
        super().__init__()
        # The original checkpoint reserves the DINO guidance width in the
        # transposed-convolution output (e.g. 128 -> 96 for 32 guidance).
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels - dino_guidance_channels,
            kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            in_channels + clip_guidance_channels, out_channels
        )
        self.clip_guidance_channels = clip_guidance_channels
        self.dino_guidance_channels = dino_guidance_channels

    def forward(self, x, clip_guidance=None, dino_guidance=None):
        x = self.up(x)
        if clip_guidance is not None:
            t = x.size(0) // clip_guidance.size(0)
            x = torch.cat([x, repeat(clip_guidance,
                "B C H W -> (B T) C H W", T=t)], dim=1)
        if dino_guidance is not None:
            t = x.size(0) // dino_guidance.size(0)
            x = torch.cat([x, repeat(dino_guidance,
                "B C H W -> (B T) C H W", T=t)], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        group1 = max(1, mid_channels // 16)
        group2 = max(1, out_channels // 16)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group1, mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group2, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
