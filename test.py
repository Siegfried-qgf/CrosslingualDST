import torch
from transformers import AutoModelForSeq2SeqLM
ckpt_para = torch.load("./ckpt/test/ckpt-epoch1/pytorch_model.bin",
                       map_location=torch.device('cpu'))

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
print(dict(model.named_parameters()).keys())