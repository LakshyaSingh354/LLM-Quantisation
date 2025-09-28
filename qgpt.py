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

model = GPT2LMHeadModel.from_pretrained("gpt2-large")

# replace first linear layer in the MLP
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        new_module = QuantLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            original_weight=module.weight.data,
            original_bias=module.bias.data if module.bias is not None else None
        )
        
        # ✅ FIX: Handle layers that are direct children of the model
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            # The parent is the model itself
            parent_module = model
            child_name = name
            
        setattr(parent_module, child_name, new_module)
        print(f"Replaced {name} with QuantLinear")
        break # Keep this to only replace the first layer

model.cuda().eval()

# --------------------
# Load WikiText-2 test split
# --------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10%]")
encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", truncation=False)

max_length = model.config.n_positions
stride = 512
input_ids = encodings.input_ids

nlls = []
for i in tqdm(range(0, input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = i + stride
    trg_len = end_loc - i
    input_ids_chunk = input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids_chunk.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"Quantized Perplexity (GPT-2 Large): {ppl.item():.2f}")


batch_size = 8
seq_len = 128
inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)

# Warmup
with torch.no_grad():
    _ = model(inputs)

# Timed run
iters = 20
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(iters):
        _ = model(inputs)
torch.cuda.synchronize()
end = time.time()

tokens_processed = batch_size * seq_len * iters
throughput = tokens_processed / (end - start)
print(f"Throughput: {throughput:.2f} tokens/sec (bs={batch_size}, seq={seq_len})")