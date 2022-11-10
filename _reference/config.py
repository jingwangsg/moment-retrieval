from .test_lazycall import SimpleClass
from detectron2.config import LazyCall as L

model_cfg = dict(x=3, y=2)

instance = [L(SimpleClass)(x="'${model_cfg.x}'", y="${model_cfg.y}")]
model_cfg.update(dict(instance=instance))
