from data.processor import *
from detectron2.config import LazyCall as L

sample_video = L(SequenceSampler)(
    axis=0, max_len="${data.video_max_len}", from_key="video_feat")
pad_video = L(SequencePad)(
    from_key="video_feat.sample", axis=0, fill_value=0.0)
pad_text = L(SequencePad)(from_key="text_feat", axis=0, fill_value=0)
rename = L(Rename)(
    from_keys=[
        "text_feat.pad",
        "text_feat.mask",
        "video_feat.sample.pad",
        "video_feat.sample.mask",
    ],
    to_keys=["text_feat", "text_mask", "vid_feat", "vid_mask"])

# text_feat, vid_feat -> text_feat, text_mask, vid_feat, vid_mask
processors = [sample_video, pad_video, pad_text, rename]