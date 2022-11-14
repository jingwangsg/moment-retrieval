import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, einsum
from kn_util.nn import clones
import numpy as np


class TimeSFormerXAttention(nn.Module):
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

    def get_attn_logits(self, feat_k, mask_k, feat_q, mask_q, mask_lv, proj):
        B, Lv, Lq, _ = feat_q.shape
        B, Lv, Lk, _ = feat_k.shape
        B, Lv = mask_lv.shape
        B, Lk = mask_k.shape
        B, Lq = mask_q.shape

        feat_q = rearrange(
            proj[0](feat_q),
            "b lv lq (h dh) -> b lv lq h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        feat_k = rearrange(
            proj[1](feat_k),
            "b lv lk (h dh) -> b lv lk h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        attn_logits = einsum(feat_q, feat_k, "b lv lq h dh, b lv lk h dh->b lv h lq lk")
        attn_mask = repeat(
            einsum(mask_q, mask_k, "b lq, b lk->b lq lk"),
            "b lq lk->b lv h lq lk",
            lv=Lv,
            h=self.num_head,
        )
        attn_logits[~attn_mask] = -10000.0

        return attn_logits  # b lv h lq lk

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, HW, _ = vid_feat.shape
        B, Lv, Lt, _ = txt_feat.shape
        B, Lv = vid_mask.shape
        B, Lt = txt_mask.shape

        v2v_value = self.v2v_proj[2](vid_feat)
        t2v_value = self.t2v_proj[2](txt_feat)
        v_value = torch.cat([v2v_value, t2v_value], dim=2)
        mask_hw = torch.ones((B, HW), device=vid_feat.device, dtype=torch.bool)
        v_value = rearrange(v_value, "b lv lk (h dh)->b lv lk h dh", h=self.num_head)

        v2t_value = self.v2t_proj[2](vid_feat)
        t2t_value = self.t2t_proj[2](txt_feat)
        t_value = torch.cat([v2t_value, t2t_value], dim=2)
        t_value = rearrange(t_value, "b lv lk (h dh)->b lv lk h dh", h=self.num_head)

        v2v_logits = self.get_attn_logits(
            vid_feat, mask_hw, vid_feat, mask_hw, vid_mask, self.v2v_proj
        )
        t2v_logits = self.get_attn_logits(
            txt_feat, txt_mask, vid_feat, mask_hw, vid_mask, self.t2v_proj
        )
        v2t_logits = self.get_attn_logits(
            vid_feat, mask_hw, txt_feat, txt_mask, vid_mask, self.v2t_proj
        )
        t2t_logits = self.get_attn_logits(
            txt_feat, txt_mask, txt_feat, txt_mask, vid_mask, self.t2t_proj
        )

        v_logits = torch.cat([v2v_logits, t2v_logits], dim=-1)
        v_logits = self.do(v_logits)
        t_logits = torch.cat([v2t_logits, t2t_logits], dim=-1)
        t_logits = self.do(t_logits)

        vid_feat = einsum(
            F.softmax(v_logits, dim=-1) / np.sqrt(self.d_head),
            v_value,
            "b lv h lq lk, b lv lk h d -> b lv lq h d",
        )
        vid_feat = rearrange(vid_feat, "b lv lq h d -> b lv lq (h d)")

        txt_feat = einsum(
            F.softmax(t_logits, dim=-1) / np.sqrt(self.d_head),
            t_value,
            "b lv h lq lk, b lv lk h d -> b lv lq h d",
        )
        txt_feat = rearrange(txt_feat, "b lv lq h d -> b lv lq (h d)")

        return vid_feat, txt_feat


class TimeSFormerXOutput(nn.Module):
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


class TimeSFormerXEncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, ff_dim, dropout) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            d_model, num_head, dropout, batch_first=True
        )
        self.temporal_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.cross_attn = TimeSFormerXAttention(
            d_model=d_model, num_head=num_head, dropout=dropout
        )
        self.output = TimeSFormerXOutput(d_model, ff_dim, dropout)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        """
        x = temporal_attn(x) + x
        x = cross_attn(x) + x
        x = OUTPUT(x)
        """
        B, Lv, HW, _ = vid_feat.shape

        vid_feat = rearrange(vid_feat, "b lv hw d -> (b hw) lv d")
        extend_vid_mask = repeat(vid_mask, "b lv -> (b i) lv", i=HW)
        vid_feat_attn, _ = self.temporal_attn(
            query=vid_feat,
            key=vid_feat,
            value=vid_feat,
            key_padding_mask=~extend_vid_mask,
        )
        vid_feat = self.temporal_ln(vid_feat + vid_feat_attn)
        vid_feat = rearrange(vid_feat, "(b hw) lv d -> b lv hw d", b=B, hw=HW)
        vid_feat_, txt_feat_ = self.cross_attn(vid_feat, vid_mask, txt_feat, txt_mask)
        vid_feat = vid_feat + vid_feat_
        txt_feat = txt_feat + txt_feat_

        vid_feat, txt_feat = self.output(vid_feat, txt_feat)

        return vid_feat, txt_feat


class TimeSFormerXEncoder(nn.Module):
    def __init__(self, layer, num_layer) -> None:
        super().__init__()
        self.layers = clones(layer, num_layer)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, HW, _ = vid_feat.shape
        txt_feat = repeat(txt_feat, "b lt d->b lv lt d", lv=Lv)
        for layer in self.layers:
            vid_feat, txt_feat = layer(vid_feat, vid_mask, txt_feat, txt_mask)

        return vid_feat, txt_feat


class TimeSFormerX(nn.Module):
    """Similar to devided space-time attention in TimeSFormer
    Except we incorporate text tokens at spatial attention stage for cross modal interaction
    """

    def __init__(
        self,
        d_model=512,
        num_head=8,
        num_layer=6,
        input_vid_dim=768,
        input_txt_dim=768,
        ff_dim=1024,
        max_num_patches=512,
        max_txt_len=30,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.vid_proj = nn.Linear(input_vid_dim, d_model)
        self.txt_proj = nn.Linear(input_txt_dim, d_model)

        self.vid_pe = nn.Embedding(max_num_patches, d_model)
        self.txt_pe = nn.Embedding(max_txt_len, d_model)
        self.type_embed_vid = nn.Parameter(torch.randn(d_model))
        self.type_embed_txt = nn.Parameter(torch.randn(d_model))
        self.vid_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.txt_ln = nn.LayerNorm(d_model, eps=1e-12)

        layer = TimeSFormerXEncoderLayer(
            d_model=d_model, num_head=num_head, ff_dim=ff_dim, dropout=dropout
        )
        self.encoder = TimeSFormerXOutput(layer, num_layer)

    def _get_embedding(self, vid_feat, txt_feat):
        B, Lv, H, W, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat = rearrange(self.vid_proj(vid_feat), "b lv h w d -> b (lv h w) d")
        vid_feat = self.vid_ln(
            vid_feat + self.vid_pe.weight[None, : vid_feat.shape[1]] + self.type_embed_vid
        )
        vid_feat = rearrange(vid_feat, "b (lv h w) d -> b lv (h w) d", lv=Lv, h=H, w=W)

        txt_feat = self.txt_proj(txt_feat)
        txt_feat = self.txt_ln(
            txt_feat + self.txt_pe.weight[None, : txt_feat.shape[1]] + self.type_embed_txt
        )

        return vid_feat, txt_feat

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, H, W, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat, txt_feat = self._get_embedding(vid_feat, txt_feat)
        vid_feat, txt_feat = self.encoder(vid_feat, vid_mask, txt_feat, txt_mask)

        vid_feat = rearrange(vid_feat, "b lv (h w) d -> b lv h w d", h=H, w=W)

        return vid_feat, txt_feat


if __name__ == "__main__":
    model = TimeSFormerX(
        d_model=512,
        num_head=16,
        num_layer=3,
        input_vid_dim=1024,
        input_txt_dim=768,
        max_num_patches=6 * 6 * 256,
        max_txt_len=31,
        dropout=0.1,
    )
    model = model.cuda()
    B = 16
    Lv = 128
    vid_feat, txt_feat = model(
        vid_feat=torch.randn((B, Lv, 6, 6, 1024), device="cuda"),
        vid_mask=torch.ones((B, Lv), dtype=torch.bool, device="cuda"),
        txt_feat=torch.randn((B, 16, 768), device="cuda"),
        txt_mask=torch.ones((B, 16), dtype=torch.bool, device="cuda"),
    )
    print(torch.cuda.max_memory_allocated("cuda") / (1024**3))
    import ipdb

    ipdb.set_trace()  # FIXME
