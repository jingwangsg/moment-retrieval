from kn_util.data import general_sample_sequence
import numpy as np

def test_general_sample_seq():
    x = np.random.randn(128, 512)
    random_sampled_x = general_sample_sequence(x, axis=0, max_len=32, mode="random")
    center_sampled_x = general_sample_sequence(x, axis=0, max_len=32, mode="center")
    max_pool_x = general_sample_sequence(x, axis=0, max_len=32, mode="maxpool")
    avg_pool_x = general_sample_sequence(x, axis=0, max_len=32, mode="avgpool")

    import ipdb; ipdb.set_trace() #FIXME

if __name__ == "__main__":
    test_general_sample_seq()