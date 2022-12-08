from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.reading_service import PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from lightning_lite.utilities import rank_zero_only as RZO
import torch
import torch.distributed as dist
import os
import datetime
from kn_util.data.datapipe import prepare_for_ddp

def initialize_ddp_from_env():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=rank,
                            timeout=datetime.timedelta(seconds=5400))

    torch.cuda.set_device(local_rank)

def divisible_to_3(x):
    return (x % 3 == 0).item()

def test_distributed_dataloader2():
    initialize_ddp_from_env()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    datapipe = IterableWrapper([torch.tensor([i]) for i in range(19)])
    datapipe = prepare_for_ddp(datapipe)

    dataloader = DataLoader2(datapipe, reading_service=PrototypeMultiProcessingReadingService(num_workers=8))

    for x in dataloader:
        # print(x)
        x = x.cuda()
        tensor_list = [torch.zeros_like(x) for _ in range(world_size)] if rank == 0 else None
        group = dist.group.WORLD
        dist.gather(x, tensor_list, group=group, dst=0)
        RZO(print)(tensor_list)

if __name__ == "__main__":
    test_distributed_dataloader2()