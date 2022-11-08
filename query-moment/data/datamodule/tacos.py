import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from kn_util.general import global_registry
from kn_util.file import load_json
from data.datamodule.base import BaseDataModule
import os.path as osp
import numpy as np


@global_registry.register_datamodule("tacos")
class TACoSDataModule(BaseDataModule):
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
                        num_frames=num_frames,
                        sample_id=video_id + f"_{idx}",
                    )

                    dataset += [elem]
            self.datasets[domain] = dataset


if __name__ == "__main__":
    import sys
    import os.path as osp

    sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
    from kn_util.config import load_config

    cfg = load_config(
        "/export/home2/kningtg/WORKSPACE/moment-retrieval/query-moment/config/common/default.yaml"
    )
    # cfg.debug = True
    datamodule = global_registry.build_datamodule("tacos", cfg=cfg)
    train_loader = datamodule.get_dataloader("train")
    from tqdm import tqdm
    for data in tqdm(train_loader):
        pass
    import ipdb

    ipdb.set_trace()  # FIXME
