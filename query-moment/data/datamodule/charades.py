import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from kn_util.general import registry
from kn_util.file import load_json
from data.datamodule.base import TSGVDataModule
import os.path as osp
import numpy as np


@registry.register_datamodule("charades")
class CharadesDataModule(TSGVDataModule):

    def load_data(self):
        self.datasets = dict()
        dataset_dir = self.cfg.data.dataset_dir

        ret_dataset = []
        for domain in ["train", "test"]:
            ret_dataset = []
            txt_file = osp.join(dataset_dir, "annot",
                                f"charades_sta_{domain}.txt")
            with open(txt_file, "r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    annot, sentence = line.split("##")
                    video_id, st, ed = annot.split()
                    text_id = f"{video_id}_{idx}"
                    cur_elem = dict(
                        video_id=video_id,
                        text=sentence,
                        text_id=text_id,
                        gt=np.array([float(st), float(ed)]))
                    ret_dataset += [cur_elem]
            self.datasets[domain] = ret_dataset

        self.datasets["val"] = self.datasets["test"]


if __name__ == "__main__":
    from detectron2.config import LazyConfig
    from pprint import pformat
    from omegaconf import OmegaConf
    cfg = LazyConfig.load("../../config/ms_temporal_detr/ms_temporal_detr.py")
    cfg.data.dataset = "charades"
    print(pformat(OmegaConf.to_container(cfg, resolve=False)))
    datamodule = CharadesDataModule(cfg)

    import ipdb
    ipdb.set_trace()  #FIXME
