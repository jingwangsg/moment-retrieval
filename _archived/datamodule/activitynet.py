import pytorch_lightning as pl
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from data.datamodule.base import TSGVDataModule
from kn_util.file import load_json
from kn_util.general import registry
import os.path as osp
import numpy as np


@registry.register_datamodule("activitynet")
class ActivityNetDataModule(TSGVDataModule):
    def load_data(self):
        self.datasets = dict()
        dataset_dir = self.cfg.data.dataset_dir
        for domain in ["train", "val", "test"]:
            dataset = []
            annots = load_json(
                osp.join(dataset_dir, "annot", f"{domain}.json"))
            for video_id, annot in annots.items():
                duration = annot["duration"]
                for idx, (timestamp, sentence) in enumerate(
                        zip(annot["timestamps"], annot["sentences"])):
                    timestamp = np.array(timestamp)
                    gt = timestamp / duration
                    elem = dict(
                        video_id=video_id,
                        gt=gt,
                        text=sentence,
                        text_id=video_id + f"_{idx}",
                    )

                    dataset += [elem]
            self.datasets[domain] = dataset


if __name__ == "__main__":
    from detectron2.config import LazyConfig
    cfg = LazyConfig.load("../../config/ms_temporal_detr/ms_temporal_detr.py")
    cfg.data.dataset = "activitynet"
    datamodule = ActivityNetDataModule(cfg)

    import ipdb; ipdb.set_trace() #FIXME