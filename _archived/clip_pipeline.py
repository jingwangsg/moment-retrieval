from kn_util.file import load_hdf5
import os.path as osp
import numpy as np

class Pipeline:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
    
    def pre_process(self, all_samples):
        pass
    
    def process(self, batch):
        pass

class TSGVPipeline(Pipeline):
    
    def get_text_feature(self, text):
        pass
    
    def get_video_feature(self, video_id):
        cfg = self.cfg
        dcfg = self.cfg.data

        if dcfg.vid_read_type == "hdf5":
            if not hasattr(self, "hdf5_vid"):
                self.hdf5_vid = osp.join(cfg.flags.dataset_dir, dcfg.vid_feat_type + ".hdf5")
            path_template = dcfg.vid_path_template
            self.hdf5_vid[path_template.format(video_id)]
        elif dcfg.vid_read_type == "npy":
            np.
    
    def pre_process(self, batch):
        return super().pre_process(batch)
    
    def process(self, all_samples):
        pass