from kn_util.general import global_registry
from kn_util.file import load_json
from .base import BaseDataModule
import os.path as osp
import numpy as np


@global_registry.register_datamodule("tacos")
class TACoSBuilder(BaseDataModule):
    def load_data(self):
        data_dir = self.cfg.data.data_dir
        for domain in ["train", "val", "test"]:
            dataset = []
            annots = load_json(osp.join(data_dir, "annot", f"{domain}.json"))
            for video_file, annot in annots.items():
                video_id = video_file[:4]
                num_frames = annot["num_frames"]
                for timestamp, sentence in zip(annot["timestamps"], annot["sentences"]):
                    timestamp = np.array(timestamp)
                    gt = timestamp / num_frames
                    elem = dict(
                        video_id=video_id, gt=gt, text=sentence, num_frames=num_frames
                    )

                    dataset += [elem]
            self.datasets[domain] = dataset