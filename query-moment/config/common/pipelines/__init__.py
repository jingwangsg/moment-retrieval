from .clip_pipeline import pipeline as clip
from .default_pipeline import pipeline as default

pipeline_dict = dict(clip=clip,
                     default=default)

def build_pipeline(name):
    return pipeline_dict[name]