import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
import math, time
from qlinear import QuantLinear
from transformers.pytorch_utils import Conv1D

from tqdm import tqdm
# --------------------
# Load model & tokenizer
# --------------------
model_name = "gpt2-large"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)
model.cuda()

layers_to_replace = []
for name, module in model.named_modules():
    # ✅ Check for both nn.Linear and the special Conv1D class
    if isinstance(module, (nn.Linear, Conv1D)):
        layers_to_replace.append(name)

print(f"Found {len(layers_to_replace)} linear layers to replace.")

# --- Step 2: Iterate over the collected names and replace them ---
for name in layers_to_replace[:3]:
    original_module = model.get_submodule(name)
    in_features = None
    # Handle attribute differences between nn.Linear and Conv1D
    if isinstance(original_module, nn.Linear):
        in_features = original_module.in_features
        out_features = original_module.out_features
        original_weight = original_module.weight.data
        original_bias = original_module.bias.data if original_module.bias is not None else None
        
    elif isinstance(original_module, Conv1D):
        # The Conv1D weight is stored transposed, so we need to adapt
        in_features = original_module.weight.shape[0]
        out_features = original_module.weight.shape[1]
        original_weight = original_module.weight.data.t() # ⚠️ Transpose the weight!
        original_bias = original_module.bias.data if original_module.bias is not None else None

    if in_features:
        # Create your new quantized module
        new_module = QuantLinear(
            in_features,
            out_features,
            bias=(original_bias is not None),
            original_weight=original_weight,
            original_bias=original_bias
        )
        
        # Get the parent module to set the new child
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_module = model
            child_name = name
            
        setattr(parent_module, child_name, new_module)
        print(f"Replaced {name} with QuantLinear")
    # break


model.eval()

initial_mem = torch.cuda.memory_allocated() / 1024**3 # Convert bytes to GB
print(f"Initial model memory usage: {initial_mem:.2f} GB")

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


batch_size = 4
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

peak_mem = torch.cuda.max_memory_allocated() / 1024**3 # Convert bytes to GB
print(f"Peak GPU memory usage during run: {peak_mem:.2f} GB")
