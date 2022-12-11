from kn_util.data import general_sample_sequence, general_pad
import numpy as np

def test_general_sample_seq():
    x = np.random.randn(128, 512)
    random_sampled_x = general_sample_sequence(x, axis=0, max_len=32, mode="random")
    center_sampled_x = general_sample_sequence(x, axis=0, max_len=32, mode="center")
    max_pool_x = general_sample_sequence(x, axis=0, max_len=32, mode="maxpool")
    avg_pool_x = general_sample_sequence(x, axis=0, max_len=32, mode="avgpool")

    import ipdb; ipdb.set_trace() #FIXME

def test_general_pad():
    x = [np.array([[0,1,3],[0,1,4]]), np.array([[0,2,4,7], [0,2,6,8]])]
    x = general_pad(x, fill_value="last", axis=1, to_length=10)
    import ipdb; ipdb.set_trace() #FIXME

if __name__ == "__main__":
    # test_general_sample_seq()
    test_general_pad()