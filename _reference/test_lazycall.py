from detectron2.config import LazyConfig, instantiate
class SimpleClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

if __name__ == "__main__":
    cfg = LazyConfig.load("./config2.py")
    instance = instantiate(cfg.model_cfg.instance)
    import ipdb; ipdb.set_trace() #FIXME