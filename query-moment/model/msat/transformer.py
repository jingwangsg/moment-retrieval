import torch
import torch.nn as nn
import torch.nn.functional as F
from kn_util.nn import clones
from einops import rearrange, einsum, repeat
import numpy as np
from kn_util.nn.layers import MLP


class VisualLinguisticAttention(nn.Module):

    def __init__(self, d_model, num_head, dropout=0.1, sr_ratio=1) -> None:
        super().__init__()
        d_head = d_model // num_head
        self.t2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.t2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.do = nn.Dropout(dropout)

        self.d_head = d_head
        self.num_head = num_head
        self.sr_ratio = sr_ratio

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
            vid_mask_sr = nn.MaxPool1d(kernel_size=self.sr_ratio, stride=self.sr_ratio)(vid_mask_sr.float())
            vid_mask_sr = (vid_mask_sr > 0)[:, 0, :]
        else:
            vid_feat_sr = vid_feat
            vid_mask_sr = vid_mask


        v2v_value = self.v2v_proj[2](vid_feat_sr)
        t2v_value = self.t2v_proj[2](txt_feat)
        v_value = torch.cat([v2v_value, t2v_value], dim=1)
        v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2t_value = self.v2t_proj[2](vid_feat_sr)
        t2t_value = self.t2t_proj[2](txt_feat)
        t_value = torch.cat([v2t_value, t2t_value], dim=1)
        t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2v_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, vid_feat, vid_mask, self.v2v_proj)
        t2v_logits = self.get_attn_logits(txt_feat, txt_mask, vid_feat, vid_mask, self.t2v_proj)
        v2t_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, txt_feat, txt_mask, self.v2t_proj)
        t2t_logits = self.get_attn_logits(txt_feat, txt_mask, txt_feat, txt_mask, self.t2t_proj)

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


class VisualLinguisticEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ff_dim, dropout) -> None:
        super().__init__()
        self.attn = VisualLinguisticAttention(d_model, num_heads, dropout)
        self.ffn_vid = MLP(input_size=d_model, hidden_size=ff_dim, output_size=d_model, activation=F.gelu, has_ln=False)
        self.ffn_txt = MLP(input_size=d_model, hidden_size=ff_dim, output_size=d_model, activation=F.gelu, has_ln=False)
        self.do = nn.Dropout(dropout)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        vid_feat, txt_feat = self.attn(vid_feat, vid_mask, txt_feat, txt_mask)
        vid_feat = vid_feat + self.do(vid_feat)
        txt_feat = txt_feat + self.do(txt_feat)

        vid_feat = self.do(self.ffn_vid(vid_feat)) + vid_feat
        txt_feat = self.do(self.ffn_txt(txt_feat)) + txt_feat

        return vid_feat, txt_feat


class VisualLinguisticTransformer(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads,
                 ff_dim,
                 num_layers,
                 dropout=0.1,
                 max_len_vid=128,
                 max_len_txt=30,
                 input_size_vid=1024,
                 input_size_txt=300) -> None:
        super().__init__()
        layer = VisualLinguisticEncoderLayer(d_model, num_heads, ff_dim, dropout)
        self.layers = clones(layer, num_layers)
        self.w_vid = nn.Linear(input_size_vid, d_model)
        self.w_txt = nn.Linear(input_size_txt, d_model)
        self.ln_vid = nn.LayerNorm(d_model, eps=1e-12)
        self.ln_txt = nn.LayerNorm(d_model, eps=1e-12)
        self.do = nn.Dropout(dropout)
        self.pe_vid = nn.Embedding(max_len_vid, d_model)
        self.pe_txt = nn.Embedding(max_len_txt, d_model)

    def embedding(self, vid_feat, txt_feat):
        vid_feat = self.w_vid(vid_feat) + self.pe_vid.weight[None, :vid_feat.shape[1]]
        vid_feat = self.do(self.ln_vid(vid_feat))

        txt_feat = self.w_txt(txt_feat) + self.pe_txt.weight[None, :txt_feat.shape[1]]
        txt_feat = self.do(self.ln_txt(txt_feat))
        return vid_feat, txt_feat

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        vid_feat, txt_feat = self.embedding(vid_feat, txt_feat)
        for layer in self.layers:
            vid_feat, txt_feat = layer(vid_feat, vid_mask, txt_feat, txt_mask)

        return vid_feat, txt_feat