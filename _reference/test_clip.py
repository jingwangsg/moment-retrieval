from transformers import AutoModel, AutoTokenizer

pretrained = "openai/clip-vit-base-patch32"
model = AutoModel.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

inputs = tokenizer("I'm your daddy!!")
o = model(**inputs)
import ipdb; ipdb.set_trace() #FIXME