# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch
import json
import os

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .RSKT_Decoder import RSKT_Decoder
from RSKT_Seg.third_party import clip
from RSKT_Seg.third_party import imagenet_templates


class RSKT_Seg_Predictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        class_descriptions_dir: str,
        num_class_descriptions: int,
        cmc_topk: int,
        cmc_tau: float,
        clip_pretrained: str,
        clip_pretrained_remote: str,
        clip_pretrained_weights_remote: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        dino_feature_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_clip_guidance_dims: list,
        decoder_clip_guidance_proj_dims:list, 
        decoder_dino_guidance_dims:list,
        decoder_dino_guidance_proj_dims:list, 
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        use_clip_corr: bool,
        use_dino_corr: bool,
        fusion_type: str,
        use_remote_clip: bool, 
        use_remote_dino: bool, 
        use_rotate: bool, 
        num_heads: int,
        num_layers: tuple,
        use_efficient: bool,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        # Class files define output channel order; description files provide anchors.
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts
        if not isinstance(self.class_texts, list) or not isinstance(self.test_class_texts, list):
            raise TypeError("TRAIN_CLASS_JSON and TEST_CLASS_JSON must contain ordered class-name lists")
        self.class_descriptions_dir = class_descriptions_dir
        self.num_class_descriptions = int(num_class_descriptions)
        self.cmc_topk = int(cmc_topk)
        self.cmc_tau = float(cmc_tau)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None

        # CLIP
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        
        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "rs_templates":
            prompt_templates = imagenet_templates.RS_ALL_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        self.train_description_tokens = self._load_description_tokens(train_class_json, self.class_texts)
        self.test_description_tokens = self._load_description_tokens(test_class_json, self.test_class_texts)
        
        transformer = RSKT_Decoder(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            clip_guidance_dim=appearance_guidance_dim,
            clip_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            decoder_clip_guidance_dims=decoder_clip_guidance_dims,
            decoder_clip_guidance_proj_dims=decoder_clip_guidance_proj_dims,
            decoder_dino_guidance_dims=decoder_dino_guidance_dims,
            decoder_dino_guidance_proj_dims=decoder_dino_guidance_proj_dims,
            use_clip_corr = use_clip_corr,
            use_dino_corr = use_dino_corr,
            fusion_type = fusion_type,
            use_remote_clip = use_remote_clip,
            use_remote_dino = use_remote_dino,
            use_rotate = use_rotate,
            num_layers=num_layers,
            use_efficient=use_efficient,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            clip_feat_dim=int(clip_model.visual.proj.shape[-1]),
            dino_feat_dim=dino_feature_dim,
            text_feat_dim=int(clip_model.text_projection.shape[1]),
            cmc_topk=self.cmc_topk,
            cmc_tau=self.cmc_tau,
            )
        self.transformer = transformer
        
        self.tokens = None
        self.cache = None

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["class_descriptions_dir"] = cfg.MODEL.SEM_SEG_HEAD.CLASS_DESCRIPTIONS_DIR
        ret["num_class_descriptions"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASS_DESCRIPTIONS
        ret["cmc_topk"] = cfg.MODEL.SEM_SEG_HEAD.CMC_TOPK
        ret["cmc_tau"] = cfg.MODEL.SEM_SEG_HEAD.CMC_TAU
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["clip_pretrained_remote"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_REMOTE
        ret["clip_pretrained_weights_remote"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_WEIGHTS_REMOTE
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["dino_feature_dim"] = cfg.MODEL.SEM_SEG_HEAD.DINO_FEATURE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS
        ret["decoder_clip_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS
        ret["decoder_clip_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_PROJ_DIMS
        ret["decoder_dino_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_DIMS
        ret["decoder_dino_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_PROJ_DIMS       
        ret['use_clip_corr'] = cfg.MODEL.SEM_SEG_HEAD.USE_CLIP_CORR
        ret['use_dino_corr'] = cfg.MODEL.SEM_SEG_HEAD.USE_DINO_CORR

        ret["use_remote_clip"] = cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP
        ret["use_remote_dino"] = cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_DINO
        ret["use_rotate"] = cfg.MODEL.SEM_SEG_HEAD.USE_ROTATE

        ret['fusion_type'] = cfg.MODEL.SEM_SEG_HEAD.FUSION_TYPE
        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["use_efficient"] = cfg.MODEL.SEM_SEG_HEAD.USE_EFFICIENT
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        return ret

    def forward(self, files_name, clip_feat, dino_feat, clip_guidance, dino_guidance, prompt=None, gt_cls=None):
        if clip_guidance is not None:
            clip_guidance = [clip_guidance[k] for k in clip_guidance.keys()][::-1]
        else:
            clip_guidance = None

        text_tokens = self.train_description_tokens if self.training else self.test_description_tokens
        if gt_cls is not None:
            indices = gt_cls if torch.is_tensor(gt_cls) else torch.as_tensor(gt_cls, dtype=torch.long)
            text_tokens = text_tokens[indices.cpu()]
        text = self.get_text_embeds(text_tokens, self.clip_model, prompt)
        if clip_feat is not None:
            if isinstance(clip_feat, list):
                text = text.repeat(clip_feat[0].shape[0], 1, 1, 1)
            else:
                text = text.repeat(clip_feat.shape[0], 1, 1, 1)
        else:
            text = text.repeat(dino_feat.shape[0], 1, 1, 1)
        # text: [B,N,1,512]
        out = self.transformer(files_name, clip_feat, dino_feat, text, clip_guidance, dino_guidance)
        return out

    @torch.no_grad()
    def _legacy_class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    

    def _legacy_get_text_embeds(self, classnames, templates, clip_model, prompt=None):
            if self.cache is not None and not self.training:
                return self.cache
            T = len(classnames)
            P = len(templates)

            if self.tokens is None or prompt is not None:
                tokens = []
                for classname in classnames:
                    if ', ' in classname:
                        classname_splits = classname.split(', ')
                        texts = [template.format(classname_splits[0]) for template in templates]
                    else:
                        texts = [template.format(classname) for template in templates]  # format with class
                    
                    if self.tokenizer is not None:
                        texts = self.tokenizer(texts).cuda()
                    else: 
                        texts = clip.tokenize(texts).cuda()
                    tokens.append(texts)
                
                tokens = torch.stack(tokens, dim=0)

                if prompt is None:
                    self.tokens = tokens
            elif self.tokens is not None and prompt is None:
                tokens = self.tokens

            tokens_flat = tokens.view(-1, 77)

            class_embeddings = clip_model.encode_text(tokens_flat, prompt)
            class_embeddings = class_embeddings.view(T, P, -1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            
            if not self.training:
                self.cache = class_embeddings
                
            return class_embeddings

    def _load_description_tokens(self, class_json, class_names):
        import glob

        dataset_name = os.path.splitext(os.path.basename(class_json))[0]
        path = os.path.join(
            self.class_descriptions_dir,
            "{}_class_descriptions_Gemini3.1_Pro.json".format(dataset_name),
        )
        if not os.path.isfile(path):
            candidates = glob.glob(os.path.join(self.class_descriptions_dir, "*_class_descriptions_Gemini3.1_Pro.json"))
            path_by_stem = {
                os.path.basename(candidate).split("_class_descriptions_Gemini3.1_Pro.json")[0].lower(): candidate
                for candidate in candidates
            }
            path = path_by_stem.get(dataset_name.lower(), path)
        if not os.path.isfile(path):
            raise FileNotFoundError("No Gemini3.1_Pro descriptions found for {}: {}".format(dataset_name, path))

        with open(path, "r", encoding="utf-8") as stream:
            descriptions = json.load(stream)
        descriptions_by_name = {str(name).strip().lower(): values for name, values in descriptions.items()}
        aliases = {
            "no-data": "background",
            "impervious surface": "impervious surfaces",
        }
        token_rows = []
        for class_name in class_names:
            key = str(class_name).strip().lower()
            key = aliases.get(key, key)
            if key not in descriptions_by_name:
                raise KeyError("Class {!r} is missing from {}".format(class_name, path))
            captions = descriptions_by_name[key]
            if not isinstance(captions, list) or len(captions) != self.num_class_descriptions:
                count = len(captions) if isinstance(captions, list) else type(captions).__name__
                raise ValueError(
                    "Expected {} descriptions for {!r} in {}, got {}".format(
                        self.num_class_descriptions, class_name, path, count
                    )
                )
            if not all(isinstance(caption, str) and caption.strip() for caption in captions):
                raise ValueError("Descriptions for {!r} in {} must be non-empty strings".format(class_name, path))
            token_rows.append(clip.tokenize(captions))
        return torch.stack(token_rows, dim=0)

    def get_text_embeds(self, tokens, clip_model, prompt=None):
        if self.cache is not None and not self.training and prompt is None:
            return self.cache
        num_classes, num_anchors, context_length = tokens.shape
        tokens = tokens.to(next(clip_model.parameters()).device).reshape(-1, context_length)
        # The local CLIP implementation exposes the standard encode_text(text) API.
        with torch.set_grad_enabled(self.training):
            embeddings = clip_model.encode_text(tokens)
        embeddings = embeddings.reshape(num_classes, num_anchors, -1)
        embeddings = F.normalize(embeddings, dim=-1)
        if not self.training and prompt is None:
            self.cache = embeddings.detach()
        return embeddings
