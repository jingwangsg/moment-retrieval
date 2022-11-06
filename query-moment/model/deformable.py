import torch
import torch.nn as nn

class Conv3dLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward()

class VideoTextEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = [nn.]

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        """
        Args:
            vid_feat (_type_): [bsz, vid_len, h, w, dim]
            vid_mask (_type_): [bsz, vid_len]
            txt_feat (_type_): [bsz, txt_len, dim]
            txt_mask (_type_): [bsz, txt_len]
        """


class DeformableMoment(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                vid_feat=None,
                vid_mask=None,
                txt_feat=None,
                txt_inds=None,
                txt_mask=None,
                **kwargs):
        has_vid_feat = vid_feat is not None and vid_mask is not None
        has_txt_feat = txt_feat is not None and txt_mask is not None
        has_txt_inds = txt_inds is not None and txt_mask is not None

        assert has_vid_feat and has_txt_feat, str(locals())

        return

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(dict(model=dict()))