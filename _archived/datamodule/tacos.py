import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from kn_util.general import registry
from kn_util.file import load_json
from data.datamodule.base import TSGVDataModule
import os.path as osp
import numpy as np


@registry.register_datamodule("tacos")
class TACoSDataModule(TSGVDataModule):
    def load_data(self):
        self.datasets = dict()
        dataset_dir = self.cfg.data.dataset_dir
        for domain in ["train", "val", "test"]:
            dataset = []
            annots = load_json(osp.join(dataset_dir, "annot", f"{domain}.json"))
            for video_file, annot in annots.items():
                video_id = video_file[:-4]
                num_frames = annot["num_frames"]
                for idx, (timestamp, sentence) in enumerate(
                    zip(annot["timestamps"], annot["sentences"])
                ):
                    timestamp = np.array(timestamp)
                    gt = timestamp / num_frames
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
    cfg.data.dataset = "tacos"
    datamodule = TACoSDataModule(cfg)

    import ipdb; ipdb.set_trace() #FIXME