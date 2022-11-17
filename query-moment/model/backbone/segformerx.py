import torch
import torch.nn as nn
import torch.nn.functional as F
from kn_util.nn import clones
from einops import einsum, rearrange, repeat, reduce
import numpy as np


class SegFormerXAttention(nn.Module):

    def __init__(self, d_model, num_head, sr_ratio=1, dropout=0.1) -> None:
        super().__init__()
        d_head = d_model // num_head
        self.t2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.t2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.do = nn.Dropout(dropout)

        if sr_ratio > 1.0:
            self.sr = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                padding=(sr_ratio - 1) // 2,
            )
        self.sr_ratio = sr_ratio

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

        vid_feat_sr = vid_feat.transpose(1, 2)
        if self.sr_ratio > 1.0:
            vid_feat_sr = self.sr(vid_feat_sr)
        vid_feat_sr = vid_feat_sr.transpose(1, 2)

        vid_mask_sr = vid_mask[:, None, :]
        vid_mask_sr = nn.MaxPool1d(
            kernel_size=self.sr_ratio, stride=self.sr_ratio)(
                vid_mask_sr.float())
        vid_mask_sr = (vid_mask_sr > 0)[:, 0, :]

        v2v_value = self.v2v_proj[2](vid_feat_sr)
        t2v_value = self.t2v_proj[2](txt_feat)
        v_value = torch.cat([v2v_value, t2v_value], dim=1)
        v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2t_value = self.v2t_proj[2](vid_feat_sr)
        t2t_value = self.t2t_proj[2](txt_feat)
        t_value = torch.cat([v2t_value, t2t_value], dim=1)
        t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2v_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, vid_feat,
                                          vid_mask, self.v2v_proj)
        t2v_logits = self.get_attn_logits(txt_feat, txt_mask, vid_feat,
                                          vid_mask, self.t2v_proj)
        v2t_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, txt_feat,
                                          txt_mask, self.v2t_proj)
        t2t_logits = self.get_attn_logits(txt_feat, txt_mask, txt_feat,
                                          txt_mask, self.t2t_proj)

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


class SegFormerXOutput(nn.Module):

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


class SegFormerXEncoderLayer(nn.Module):

    def __init__(self, d_model, num_head, ff_dim, sr_ratio, dropout) -> None:
        super().__init__()
        self.cross_attn = SegFormerXAttention(
            d_model=d_model,
            num_head=num_head,
            sr_ratio=sr_ratio,
            dropout=dropout)

        self.output = SegFormerXOutput(d_model, ff_dim, dropout)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        """
        x = temporal_attn(x) + x
        x = cross_attn(x) + x
        x = OUTPUT(x)
        """
        B, Lv, _ = vid_feat.shape

        vid_feat_, txt_feat_ = self.cross_attn(vid_feat, vid_mask, txt_feat,
                                               txt_mask)
        vid_feat = vid_feat + vid_feat_
        txt_feat = txt_feat + txt_feat_

        vid_feat, txt_feat = self.output(vid_feat, txt_feat)

        return vid_feat, txt_feat


class SegFormerXEncoder(nn.Module):

    def __init__(self, d_model_in, d_model_lvls, num_head_lvls, sr_ratio_lvls,
                 ff_dim_lvls, dropout) -> None:
        super().__init__()
        assert (len(d_model_lvls) == len(num_head_lvls) == len(sr_ratio_lvls)
                == len(ff_dim_lvls))
        self.layers = nn.ModuleList([
            SegFormerXEncoderLayer(
                d_model=d_model,
                num_head=num_head,
                ff_dim=ff_dim,
                sr_ratio=sr_ratio,
                dropout=dropout,
            ) for d_model, num_head, sr_ratio, ff_dim in zip(
                d_model_lvls, num_head_lvls, sr_ratio_lvls, ff_dim_lvls)
        ])

        d_model_lvls_ = [d_model_in] + d_model_lvls
        self.txt_lvl_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model_lvls_[i - 1], d_model_lvls_[i]),
                nn.LayerNorm(d_model_lvls_[i], eps=1e-12),
            ) for i in range(1, len(d_model_lvls_))
        ])
        self.patch_merge = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model_lvls_[i - 1],
                out_channels=d_model_lvls_[i],
                kernel_size=3,
                stride=2,
                padding=1,
            ) for i in range(1, len(d_model_lvls_))
        ])

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, _ = vid_feat.shape
        intermediate_states = []
        intermediate_masks = []
        for idx, layer in enumerate(self.layers):
            vid_feat = vid_feat.transpose(1, 2)
            vid_feat = self.patch_merge[idx](vid_feat)
            vid_feat = vid_feat.transpose(1, 2)
            intermediate_states += [vid_feat]
            vid_mask = vid_mask[:, None, :]
            vid_mask = nn.MaxPool1d(kernel_size=2, stride=2)(vid_mask.float())
            vid_mask = vid_mask > 0
            vid_mask = vid_mask[:, 0, :]
            intermediate_masks += [vid_mask]
            txt_feat = self.txt_lvl_projs[idx](txt_feat)

            vid_feat, txt_feat = layer(vid_feat, vid_mask, txt_feat, txt_mask)

        return intermediate_states, intermediate_masks


class SegFormerX(nn.Module):
    """similar to VisualLinguisticBert in MSAT (Multi-Stage Aggregation Transformer)"""

    def __init__(
        self,
        d_model_in=128,
        d_model_lvls=[128, 256, 512, 1024],
        num_head_lvls=[2, 4, 8, 16],
        ff_dim_lvls=[256, 512, 1024, 2048],
        sr_ratio_lvls=[8, 4, 2, 1],
        input_vid_dim=768,
        input_txt_dim=768,
        max_vid_len=256,
        max_txt_len=30,
        dropout=0.1,
        pe="learn",
        pe_kernel_size=3,
    ) -> None:
        super().__init__()

        self.vid_proj = nn.Linear(input_vid_dim, d_model_in)
        self.txt_proj = nn.Linear(input_txt_dim, d_model_in)

        self.pe = pe
        if pe == "learn":
            self.vid_pe = nn.Embedding(max_vid_len, d_model_in)
        elif pe == "conv":
            self.conv_pe = nn.Conv1d(
                in_channels=d_model_in,
                out_channels=d_model_in,
                kernel_size=pe_kernel_size,
                padding=(pe_kernel_size - 1) // 2,
            )
        else:
            raise NotImplementedError()
        self.txt_pe = nn.Embedding(max_txt_len, d_model_in)
        self.type_embed_vid = nn.Parameter(torch.randn(d_model_in))
        self.type_embed_txt = nn.Parameter(torch.randn(d_model_in))
        self.vid_ln = nn.LayerNorm(d_model_in, eps=1e-12)
        self.txt_ln = nn.LayerNorm(d_model_in, eps=1e-12)

        self.encoder = SegFormerXEncoder(
            d_model_in=d_model_in,
            d_model_lvls=d_model_lvls,
            num_head_lvls=num_head_lvls,
            sr_ratio_lvls=sr_ratio_lvls,
            ff_dim_lvls=ff_dim_lvls,
            dropout=dropout,
        )

    def _get_embedding(self, vid_feat, txt_feat):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat = self.vid_proj(vid_feat)
        txt_feat = self.txt_proj(txt_feat)

        if self.pe == "learn":
            pe = self.vid_pe.weight[None, :vid_feat.shape[1]]
        elif self.pe == "conv":
            pe = self.conv_pe(vid_feat.transpose(1, 2))
            pe = pe.transpose(1, 2)
        vid_feat = self.vid_ln(vid_feat + pe + self.type_embed_vid)

        txt_feat = self.txt_ln(txt_feat +
                               self.txt_pe.weight[None, :txt_feat.shape[1]] +
                               self.type_embed_txt)

        return vid_feat, txt_feat

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat, txt_feat = self._get_embedding(vid_feat, txt_feat)
        intermidiate_states, intermidiate_masks = self.encoder(
            vid_feat, vid_mask, txt_feat, txt_mask)

        return intermidiate_states, intermidiate_masks


if __name__ == "__main__":
    num_layer = 3
    model = SegFormerX(
        d_model_in=1024,
        d_model_lvls=[1024, 1024, 1024, 1024],
        ff_dim_lvls=[2048, 2048, 2048, 2048],
        sr_ratio_lvls=[8, 4, 2, 1],
        input_vid_dim=1024,
        input_txt_dim=768,
        max_vid_len=2048,
        max_txt_len=31,
        dropout=0.1,
        pe="conv")
    from kn_util.debug import capture_forward_and_print, explore_content
    from functools import partial

    model = model.cuda()
    for i in range(num_layer):
        model.encoder.layers[i].forward = capture_forward_and_print(
            name=f"EncoderLayer{i}")(
                model.encoder.layers[i].forward)

    B = 16
    Lv = 2048
    vid_mask = torch.ones((B, Lv), dtype=torch.bool, device="cuda")
    vid_mask[:, 900:] = False
    intermediate_states, intermediate_masks = model(
        vid_feat=torch.randn((B, Lv, 1024), device="cuda"),
        vid_mask=vid_mask,
        txt_feat=torch.randn((B, 16, 768), device="cuda"),
        txt_mask=torch.ones((B, 16), dtype=torch.bool, device="cuda"),
    )
    print(explore_content(intermediate_states, name="intermediate_states"))
    print(explore_content(intermediate_masks, name="intermediate_masks"))
    print(torch.cuda.max_memory_allocated("cuda") / (1024**3))
    import ipdb

    ipdb.set_trace()  # FIXME
