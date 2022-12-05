from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from torch.utils.data import DataLoader, RandomSampler
from torchdata.dataloader2.reading_service import MultiProcessingReadingService
from torch.utils.data import functional_datapipe
from torchdata.dataloader2 import DataLoader2
import numpy as np
from kn_util.data import general_pad
from operator import itemgetter
import h5py
from collections import OrderedDict
import copy


@functional_datapipe("parse_tacos")
class TACoSParser(IterDataPipe):

    def __init__(self, src_pipeline) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline

    def __iter__(self):
        for json_data in self.src_pipeline:
            for video_id, annot in json_data.items():
                for sentence, timestamp in zip(annot["sentences"], annot["timestamps"]):
                    gt = np.array(timestamp) / annot["num_frames"]
                    yield dict(video_id=video_id[:-4], gt=gt, sentence=sentence)


@functional_datapipe("pad_sequence")
class SequencePadder(IterDataPipe):

    def __init__(self, src_pipeline, from_key=None, return_mask=True, **pad_args) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.pad_args = pad_args
        self.return_mask = return_mask
        self.from_key = from_key

    def __iter__(self):
        for x in self.src_pipeline:
            data = x[self.from_key]
            if self.return_mask:
                padded_data, mask = general_pad(data, return_mask=self.return_mask, **self.pad_args)
                yield padded_data, mask
                x.update({self.from_key + ".pad": padded_data, self.from_Key + ".mask": mask})
            else:
                padded_data = general_pad(data, return_mask=self.return_mask, **self.pad_args)
                yield padded_data
                x.update({self.from_key + ".pad": padded_data})


@functional_datapipe("load_hdf5")
class HDF5Loader(IterDataPipe):

    def __init__(self, src_pipeline, hdf5_file, from_key) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.hdf5_file = hdf5_file
        self.from_key = from_key

    def __iter__(self):
        hdf5_handler = h5py.File(self.hdf5_file, "r")
        for x in self.src_pipeline:
            cur_id = x[self.from_key]
            x[self.from_key + ".hdf5"] = np.array(hdf5_handler[cur_id])
            yield x
        hdf5_handler.close()

@functional_datapipe("collect")
class Collect(IterDataPipe):
    def __init__(self, src_pipeline, from_keys=[], to_keys=[]) -> None:
        super().__init__()
        self.from_keys = from_keys
        self.to_keys = to_keys
    
    def __iter__(self):
        for x in self.src_pipeline:
            ret_dict = dict()
            for k, to_k in zip(self.from_keys, self.to_keys):
                ret_dict[to_k] = x[k]
            yield ret_dict
            


# @functional_datapipe("dict_zip")
# class DictZipper(IterDataPipe):
#     def __init__(self, src_pipeline, **pipelines) -> None:
#         super().__init__()
#         self.src_pipeline = src_pipeline
#         self.pipelines = pipelines

#     def __iter__(self):
#         pipelines = copy.deepcopy(self.pipelines)
#         iters = {k: iter(pipeline) for k, pipeline in pipelines.items()}
#         for x in self.src_pipeline:
#             for k in list(iters.keys()):
#                 v = next(iters[k])
#                 x[k] = v
#             yield x


def build_datapipe():
    train_json = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw/tacos/annot/train.json"
    vid_hdf5_file = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw/tacos/i3d.hdf5"
    json_file_dp = IterableWrapper([train_json])
    # parsed_json_dp = json_file_dp.open_files(mode="r", encoding="utf-8").parse_json_files().map(lambda x: x[1])
    # ! lambda cannot be pickled
    parsed_json_dp = json_file_dp.open_files(mode="r", encoding="utf-8").parse_json_files().map(itemgetter(1))

    tacos_dp = parsed_json_dp.parse_tacos().in_memory_cache().shuffle()
    samples_dp = tacos_dp.load_hdf5(hdf5_file=vid_hdf5_file, from_key="video_id")

    batch_dp = samples_dp.batch(16)
    batch_dp = batch_dp.rows2columnar()

    vid_feat = batch_dp.pad_sequence(fill_value=0.0, axis=0, from_key="video_id.hdf5")

    return vid_feat


if __name__ == "__main__":
    dp = build_datapipe()
    dataloader = DataLoader2(dp, reading_service=MultiProcessingReadingService(num_workers=8, prefetch_factor=5))

    for x in dataloader:
        print(x)
        import ipdb
        ipdb.set_trace()  #FIXME
