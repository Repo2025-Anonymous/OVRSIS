import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from .EfficientAggregator import EffAggregatorLayer
from .OriAggregator import OriAggregatorLayer
from .RSKT_Upsample import RSKT_Upsample
from .visualize_corr import visualize_corr

class CMC_Module(nn.Module):
    """Image-conditioned convex combinations of text anchors (paper Sec. 3.2)."""
    def __init__(self, clip_dim, dino_dim, text_dim, topk=5, tau=10.0):
        super().__init__()
        # The released ViT-B checkpoint uses 512-dimensional CLIP and RS-DINO
        # features.  Identity projections keep its parameter names/shapes
        # intact; learned projections remain available for other backbones.
        self.clip_projection = nn.Identity() if clip_dim == text_dim else nn.Conv2d(clip_dim, text_dim, kernel_size=1)
        self.dino_projection = nn.Identity() if dino_dim == text_dim else nn.Conv2d(dino_dim, text_dim, kernel_size=1)
        self.topk = int(topk)
        self.tau = float(tau)

    def forward(self, clip_feats, dino_feats, text_feats):
        clip_feats = self.clip_projection(clip_feats)
        dino_feats = self.dino_projection(dino_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        dino_unit = F.normalize(dino_feats, dim=1)
        scores = torch.einsum('bdhw,btpd->btphw', dino_unit, text_feats)
        global_scores = scores.mean(dim=(-1, -2))
        count = min(max(self.topk, 1), text_feats.shape[2])
        selected = global_scores.topk(count, dim=2).indices
        selected_mask = torch.zeros_like(global_scores, dtype=torch.bool)
        selected_mask.scatter_(2, selected, True)
        selected_mask = selected_mask[..., None, None]

        weights = F.softmax((scores * self.tau).masked_fill(~selected_mask, float('-inf')), dim=2)
        prototypes = torch.einsum('btphw,btpd->btdhw', weights, text_feats)
        prototypes = F.normalize(prototypes, dim=2)
        return prototypes, clip_feats, dino_feats


class RSKT_Decoder(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        clip_guidance_dim=512,
        clip_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        decoder_clip_guidance_dims=(256, 128),
        decoder_clip_guidance_proj_dims=(32, 16),
        decoder_dino_guidance_dims=(256, 128),
        decoder_dino_guidance_proj_dims=(32, 16),
        clip_feat_dim = 768,
        dino_feat_dim = 768,
        use_clip_corr = True,
        use_dino_corr = True,
        fusion_type = "simple_separate",
        use_remote_clip = True,
        use_remote_dino = True,
        use_rotate = True,
        num_layers=4,
        use_efficient=True,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
        use_cccm=True,
        text_feat_dim=768,
        cmc_topk=5,
        cmc_tau=10.0,
    ) -> None:

        super().__init__()
        # self.num_layers = num_layers
        # self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        # self.use_remote_clip = use_remote_clip
        # self.use_remote_dino = use_remote_dino
        self.use_cccm = use_cccm

        self.cmc = CMC_Module(clip_feat_dim, dino_feat_dim, text_feat_dim, cmc_topk, cmc_tau)

        if use_efficient:
            self.layers = nn.ModuleList([
                EffAggregatorLayer(hidden_dim=hidden_dim, num_heads=nheads) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                OriAggregatorLayer(
                    hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, clip_guidance=clip_guidance_proj_dim, 
                    nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
                ) for _ in range(num_layers)
            ])

        self.clip_guidance_projection = nn.Conv2d(
            clip_guidance_dim, clip_guidance_proj_dim, kernel_size=3, stride=1, padding=1
        ) if clip_guidance_dim > 0 else None
        
        self.dino_guidance_projection = nn.Conv2d(
            dino_feat_dim, clip_guidance_proj_dim, kernel_size=3, stride=1, padding=1
        ) if dino_feat_dim > 0 else None
        text_projection_input_dim = text_guidance_dim * 2 if text_guidance_dim == 512 else text_guidance_dim
        self.text_guidance_projection = nn.Linear(
            text_projection_input_dim, text_guidance_proj_dim
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
            ) for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
        ]) if decoder_clip_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
            ) for d, dp in zip(decoder_guidance_dims, decoder_dino_guidance_proj_dims)
        ]) if decoder_dino_guidance_dims[0] > 0 else None

        effective_prompt_channel = 1
        
        if fusion_type=='simple_concatenation':
            self.conv1 = nn.Conv2d(effective_prompt_channel * 2, hidden_dim, kernel_size=7, stride=1, padding=3)
        elif fusion_type=='simple_mean':
            self.conv1 = nn.Conv2d(effective_prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        elif fusion_type=='simple_separate':
            self.conv1 = nn.Conv2d(effective_prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3) 
            self.conv2 = nn.Conv2d(effective_prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
            self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3) 

        self.Fusiondecoder1 = RSKT_Upsample(
        hidden_dim, 
        decoder_dims[0],
        decoder_clip_guidance_proj_dims[0],
        decoder_dino_guidance_proj_dims[0],
        use_remote_clip = use_remote_clip,
        use_remote_dino = use_remote_dino)

        self.Fusiondecoder2 = RSKT_Upsample(
        decoder_dims[0], 
        decoder_dims[1], 
        decoder_clip_guidance_proj_dims[1], 
        decoder_dino_guidance_proj_dims[1],
        use_remote_clip = use_remote_clip,
        use_remote_dino = use_remote_dino)

        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        
    def correlation(self, visual_feats, prototypes):
        visual_feats = F.normalize(visual_feats, dim=1)
        prototypes = F.normalize(prototypes, dim=2)
        return torch.einsum('bdhw,btdhw->bthw', visual_feats, prototypes).unsqueeze(1)

    def simple_separate_corr(self, clip_corr, dino_corr, files_name):
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')

        clip_embed_corr = self.sigmoid(self.conv1(clip_corr))
        dino_embed_corr = self.sigmoid(self.conv2(dino_corr))
        fused_corr = torch.cat([clip_embed_corr, dino_embed_corr], dim = 1)
        fused_corr = self.sigmoid(self.fusion_corr(fused_corr))
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)

        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr

    def simple_concatenation_corr(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')

        fused_corr = self.sigmoid(self.conv1(torch.cat([clip_corr,dino_corr],dim = 1)))
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr 

    def simple_mean_corr(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()

        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = self.sigmoid(self.conv1(torch.mean(torch.cat([clip_corr,dino_corr],dim=1), dim = 1, keepdim=True)))
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr 

    def Fusion_conv_decoer(self, x, clip_guidance, dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0], dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1], dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, files_name, clip_feats, dino_feats, text_feats, clip_guidance, dino_guidance):
        prototypes, clip_projected, dino_projected = self.cmc(clip_feats, dino_feats, text_feats)
        clip_corr = self.correlation(clip_projected, prototypes)
        dino_corr = self.correlation(dino_projected, prototypes)

        if self.fusion_type == 'simple_separate':
            fused_corr_embed, clip_embed_corr, dino_embed_corr = self.simple_separate_corr(clip_corr=clip_corr, dino_corr=dino_corr,files_name=files_name)
            # Preserve the released RSKT fusion behavior: the learned fused
            # volume receives the CLIP branch as a residual signal.
            fused_corr_embed = fused_corr_embed + clip_embed_corr

        elif self.fusion_type == 'simple_concatenation':
            fused_corr_embed = self.simple_concatenation_corr(clip_corr, dino_corr)

        elif self.fusion_type == 'simple_mean':
            fused_corr_embed = self.simple_mean_corr(clip_corr, dino_corr)

        clip_projected_guidance = self.clip_guidance_projection(clip_guidance[0])
        dino_projected_guidance = self.dino_guidance_projection(dino_feats)
        visual_guidance = clip_projected_guidance + dino_projected_guidance
        clip_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, clip_guidance[1:])]
        dino_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # Some released ViT-B runs used the two-stream text guidance
            # (CLIP text concatenated with its normalized counterpart).
            in_dim = self.text_guidance_projection.in_features
            if text_feats.shape[-1] != in_dim:
                if text_feats.shape[-1] * 2 == in_dim:
                    text_feats = torch.cat([text_feats, text_feats], dim=-1)
                else:
                    raise ValueError("text guidance dimension does not match decoder projection")
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, visual_guidance, projected_text_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, clip_projected_decoder_guidance, dino_projected_decoder_guidance)

        return logit
