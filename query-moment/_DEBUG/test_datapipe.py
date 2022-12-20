from torchdata.datapipes.iter import IterableWrapper
dataset_dp = IterableWrapper([0,1,2,3,4,5,6,7,8])
dataset_dp.sharding_filter