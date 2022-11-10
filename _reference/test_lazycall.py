from detectron2.config import LazyConfig
class SimpleClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

if __name__ == "__main__":
    cfg = LazyConfig.load("./config2.py")
    import ipdb; ipdb.set_trace() #FIXME