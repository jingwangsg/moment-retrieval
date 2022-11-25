import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SRAttention(nn.Module):

    def __init__(self, d_model, num_head, sr_ratio, dropout) -> None:
        super().__init__()
        self.mhattn = nn.MultiheadAttention(d_model, num_head, dropout, batch_first=True)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                Rearrange("b lv d -> b d lv"),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=sr_ratio, stride=sr_ratio),
                Rearrange("b d lv -> b lv d"))
        self.sr_ratio = sr_ratio
        self.num_head = num_head

    def forward(self, vid_feat, vid_mask):
        if self.sr_ratio > 1:
            vid_feat_sr = self.sr(vid_feat)
            vid_mask_sr = F.interpolate(vid_mask[:, None, :].float(), size=vid_feat_sr.shape[1]).bool()
            vid_mask_sr = vid_mask_sr[:, 0, :]
        else:
            vid_feat_sr = vid_feat
            vid_mask_sr = vid_mask

        attn_mask = vid_mask[:, :, None] & vid_mask_sr[:, None, :]
        attn_mask = repeat(attn_mask, "b lq lk -> (b nh) lq lk", nh=self.num_head)  # make native module happy
        context, _ = self.mhattn(vid_feat, vid_feat_sr, vid_feat_sr, need_weights=False, attn_mask=attn_mask)
        return context


class BasicBlock(nn.Module):

    def __init__(self, d_model, num_head, ff_dim, sr_ratio, dropout) -> None:
        super().__init__()
        self.sr_attn = SRAttention(d_model, num_head, sr_ratio, dropout)
        self.patch_merge = nn.Sequential(Rearrange("b lv d -> b d lv"), nn.Conv1d(d_model, d_model, 3, 2, 1),
                                         Rearrange("b d lv -> b lv d"))
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model),
                                 nn.LayerNorm(d_model, eps=1e-12), nn.Dropout(dropout))

    def forward(self, vid_feat, vid_mask):
        vid_feat = self.sr_attn(vid_feat, vid_mask)
        vid_feat = self.patch_merge(vid_feat)
        vid_mask = F.interpolate(vid_mask[:, None, :].float(), size=vid_feat.shape[1]).bool()
        vid_mask = vid_mask[:, 0, :]
        vid_feat = self.ffn(vid_feat)

        return vid_feat, vid_mask


class TemporalTransformer(nn.Module):
    """
    offer pure temporal attention for vision CLIP future
    hierarchical design to achieve combined effect with decoder
    """

    def __init__(self,
                 d_model=1024,
                 num_head=8,
                 ff_dim=2048,
                 sr_ratio_lvls=[8, 4, 2, 1],
                 input_vid_dim=768,
                 max_vid_len=1024,
                 dropout=0.1) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_vid_len, d_model)
        self.vid_proj = nn.Linear(input_vid_dim, d_model)
        self.proj_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.blocks = nn.ModuleList([
            BasicBlock(d_model, num_head=num_head, ff_dim=ff_dim, sr_ratio=sr_ratio, dropout=dropout)
            for sr_ratio in sr_ratio_lvls
        ])

    def get_embedding(self, vid_feat):
        vid_feat = self.vid_proj(vid_feat) + self.pe.weight[:vid_feat.shape[1]]
        vid_feat = self.proj_ln(vid_feat)
        return vid_feat

    def forward(self, vid_feat, vid_mask):
        B, Lv, _ = vid_feat.shape
        intermediate_hiddens, intermediate_masks = [], []
        vid_feat = self.get_embedding(vid_feat=vid_feat)
        for block in self.blocks:
            vid_feat, vid_mask = block(vid_feat, vid_mask)
            intermediate_hiddens += [vid_feat]
            intermediate_masks += [vid_mask]

        return intermediate_hiddens, intermediate_masks


if __name__ == "__main__":
    B = 16
    Lv = 1024
    D = 768
    model = TemporalTransformer(input_vid_dim=D, max_vid_len=Lv)
    model = model.cuda()
    vid_feat = torch.randn((B, Lv, D), device="cuda")
    vid_mask = torch.ones((B, Lv), dtype=torch.bool, device="cuda")
    hiddens, masks = model(vid_feat, vid_mask)
    from kn_util.debug import explore_content
    print(explore_content(hiddens, "hiddens"))
    print(explore_content(masks, "masks"))

    import ipdb
    ipdb.set_trace()  #FIXME
