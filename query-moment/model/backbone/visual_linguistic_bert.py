import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, einsum
from kn_util.nn import clones
import numpy as np


class VideoTextAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout) -> None:
        super().__init__()
        d_head = d_model // num_head
        self.t2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.t2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.do = nn.Dropout(dropout)

        self.d_head = d_head
        self.num_head = num_head

    def get_attn_logits(self, feat_k, mask_k, feat_q, mask_q, proj):
        B, Lq, _ = feat_q.shape
        B, Lk, _ = feat_k.shape
        B, Lk = mask_k.shape
        B, Lq = mask_q.shape

        feat_q = rearrange(
            proj[0](feat_q),
            "b lq (h dh) -> b lq h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        feat_k = rearrange(
            proj[1](feat_k),
            "b lk (h dh) -> b lk h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        attn_logits = einsum(feat_q, feat_k, "b lq h dh, b lk h dh->b h lq lk")
        attn_mask = repeat(
            einsum(mask_q, mask_k, "b lq, b lk->b lq lk"),
            "b lq lk->b h lq lk",
            h=self.num_head,
        )
        attn_logits[~attn_mask] = -10000.0

        return attn_logits  # b lv h lq lk

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape
        B, Lv = vid_mask.shape
        B, Lt = txt_mask.shape

        v2v_value = self.v2v_proj[2](vid_feat)
        t2v_value = self.t2v_proj[2](txt_feat)
        v_value = torch.cat([v2v_value, t2v_value], dim=1)
        v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2t_value = self.v2t_proj[2](vid_feat)
        t2t_value = self.t2t_proj[2](txt_feat)
        t_value = torch.cat([v2t_value, t2t_value], dim=1)
        t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2v_logits = self.get_attn_logits(
            vid_feat, vid_mask, vid_feat, vid_mask, self.v2v_proj
        )
        t2v_logits = self.get_attn_logits(
            txt_feat, txt_mask, vid_feat, vid_mask, self.t2v_proj
        )
        v2t_logits = self.get_attn_logits(
            vid_feat, vid_mask, txt_feat, txt_mask, self.v2t_proj
        )
        t2t_logits = self.get_attn_logits(
            txt_feat, txt_mask, txt_feat, txt_mask, self.t2t_proj
        )

        v_logits = torch.cat([v2v_logits, t2v_logits], dim=-1)
        v_logits = self.do(v_logits)
        t_logits = torch.cat([v2t_logits, t2t_logits], dim=-1)
        t_logits = self.do(t_logits)

        vid_feat = einsum(
            F.softmax(v_logits, dim=-1) / np.sqrt(self.d_head),
            v_value,
            "b h lq lk, b lk h d -> b lq h d",
        )
        vid_feat = rearrange(vid_feat, "b lq h d -> b lq (h d)")

        txt_feat = einsum(
            F.softmax(t_logits, dim=-1) / np.sqrt(self.d_head),
            t_value,
            "b h lq lk, b lk h d -> b lq h d",
        )
        txt_feat = rearrange(txt_feat, "b lq h d -> b lq (h d)")

        return vid_feat, txt_feat


class VideoTextOutput(nn.Module):
    def __init__(self, d_model, ff_dim, dropout) -> None:
        super().__init__()
        self.ffn_vid = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln_vid = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn_txt = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln_txt = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, vid_feat, txt_feat):
        vid_feat = self.ln_vid(self.ffn_vid(vid_feat) + vid_feat)
        txt_feat = self.ln_txt(self.ffn_txt(txt_feat) + txt_feat)
        return vid_feat, txt_feat


class VideoTextEncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, ff_dim, dropout) -> None:
        super().__init__()
        self.cross_attn = VideoTextAttention(
            d_model=d_model, num_head=num_head, dropout=dropout
        )
        self.output = VideoTextOutput(d_model, ff_dim, dropout)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        """
        x = temporal_attn(x) + x
        x = cross_attn(x) + x
        x = OUTPUT(x)
        """
        B, Lv, _ = vid_feat.shape

        vid_feat_, txt_feat_ = self.cross_attn(vid_feat, vid_mask, txt_feat, txt_mask)
        vid_feat = vid_feat + vid_feat_
        txt_feat = txt_feat + txt_feat_

        vid_feat, txt_feat = self.output(vid_feat, txt_feat)

        return vid_feat, txt_feat


class VideoTextEncoder(nn.Module):
    def __init__(self, layer, num_layer) -> None:
        super().__init__()
        self.layers = clones(layer, num_layer)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, _ = vid_feat.shape
        for layer in self.layers:
            vid_feat, txt_feat = layer(vid_feat, vid_mask, txt_feat, txt_mask)

        return vid_feat, txt_feat


class VisualLinguisticBert(nn.Module):
    """similar to VisualLinguisticBert in MSAT (Multi-Stage Aggregation Transformer)"""

    def __init__(
        self,
        d_model=512,
        num_head=8,
        num_layer=6,
        input_vid_dim=768,
        input_txt_dim=768,
        ff_dim=1024,
        max_vid_len=256,
        max_txt_len=30,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.vid_proj = nn.Linear(input_vid_dim, d_model)
        self.txt_proj = nn.Linear(input_txt_dim, d_model)

        self.vid_pe = nn.Embedding(max_vid_len, d_model)
        self.txt_pe = nn.Embedding(max_txt_len, d_model)
        self.type_embed_vid = nn.Parameter(torch.randn(d_model))
        self.type_embed_txt = nn.Parameter(torch.randn(d_model))
        self.vid_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.txt_ln = nn.LayerNorm(d_model, eps=1e-12)

        layer = VideoTextEncoderLayer(
            d_model=d_model, num_head=num_head, ff_dim=ff_dim, dropout=dropout
        )
        self.encoder = VideoTextEncoder(layer, num_layer)

    def _get_embedding(self, vid_feat, txt_feat):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat = self.vid_proj(vid_feat)
        txt_feat = self.txt_proj(txt_feat)

        vid_feat = self.vid_ln(
            vid_feat + self.vid_pe.weight[None, : vid_feat.shape[1]] + self.type_embed_vid
        )

        txt_feat = self.txt_ln(
            txt_feat + self.txt_pe.weight[None, : txt_feat.shape[1]] + self.type_embed_txt
        )

        return vid_feat, txt_feat

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat, txt_feat = self._get_embedding(vid_feat, txt_feat)
        vid_feat, txt_feat = self.encoder(vid_feat, vid_mask, txt_feat, txt_mask)

        return vid_feat, txt_feat


if __name__ == "__main__":
    num_layer = 3
    model = VisualLinguisticBert(
        d_model=512,
        num_head=16,
        num_layer=num_layer,
        input_vid_dim=1024,
        input_txt_dim=768,
        max_vid_len=256,
        max_txt_len=31,
        dropout=0.1,
    )
    from kn_util.debug import capture_output, explore_content
    from functools import partial

    model = model.cuda()
    for i in range(num_layer):
        model.encoder.layers[i].forward = capture_output(
            partial(explore_content, name=f"EncoderLayer{i}", print_str=True)
        )(model.encoder.layers[i].forward)
    B = 16
    Lv = 256
    vid_feat, txt_feat = model(
        vid_feat=torch.randn((B, Lv, 1024), device="cuda"),
        vid_mask=torch.ones((B, Lv), dtype=torch.bool, device="cuda"),
        txt_feat=torch.randn((B, 16, 768), device="cuda"),
        txt_mask=torch.ones((B, 16), dtype=torch.bool, device="cuda"),
    )
    print(torch.cuda.max_memory_allocated("cuda") / (1024**3))
    import ipdb

    ipdb.set_trace()  # FIXME
