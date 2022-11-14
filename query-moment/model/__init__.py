from detectron2.config import instantiate

def build(cfg):
    return instantiate(cfg.model_cfg.model)