import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
import math, time
from qlinear import QuantLinear

from tqdm import tqdm
# --------------------
# Load model & tokenizer
# --------------------
model_name = "gpt2-large"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)
model.cuda()

for name,module in model.named_modules():
    print(f"{name}: {module}({type(module)})")