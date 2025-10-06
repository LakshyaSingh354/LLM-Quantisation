import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
from qlinear import QuantLinear, QuantLinearDequant
from transformers.pytorch_utils import Conv1D
import time

model_name = "gpt2-large"
device = "cuda"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

# Collect all Linear / Conv1D layers
layers_to_replace = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, Conv1D)):
        layers_to_replace.append(name)
print(f"Found {len(layers_to_replace)} linear layers to replace.")

# Load calibration ranges for QuantLinear
print("Loading calibration ranges from 'calibration_ranges.pt'...")
calibration_ranges = torch.load("calibration_ranges.pt")

print("\nStarting hybrid replacement...")

for i, name in enumerate(layers_to_replace):
    orig_mod = model.get_submodule(name)

    # Determine input/output features and weight
    if isinstance(orig_mod, nn.Linear):
        in_features = orig_mod.in_features
        out_features = orig_mod.out_features
        w = orig_mod.weight.data.contiguous()
        b = orig_mod.bias.data if orig_mod.bias is not None else None
    elif isinstance(orig_mod, Conv1D):
        # Conv1D in GPT-2: weight shape [in, out]
        in_features, out_features = orig_mod.weight.shape
        w = orig_mod.weight.data.contiguous()
        b = orig_mod.bias.data if orig_mod.bias is not None else None

    # Get calibration data for this layer
    if name in calibration_ranges:
        calib_data = calibration_ranges[name]
        calib_min = calib_data['min'].to(device)
        calib_max = calib_data['max'].to(device)
    else:
        print(f"Warning: No calibration data found for {name}, using default ranges")
        calib_min = torch.tensor(-1.0, device=device)
        calib_max = torch.tensor(1.0, device=device)

    if i < 5:  # First five layers use QuantLinear
        # Create QuantLinear with calibration ranges
        new_mod = QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=b is not None,
            original_weight=w.to(device),
            original_bias=b.to(device) if b is not None else None,
            calib_min_t=calib_min,
            calib_max_t=calib_max
        )
        layer_type = "QuantLinear"
    else:  # Rest use QuantLinearDequant
        # Create QuantLinearDequant (weight-only quantization for better accuracy)
        new_mod = QuantLinearDequant(
            in_features=in_features,
            out_features=out_features,
            bias=b is not None,
            original_weight=w.to(device),
            original_bias=b.to(device) if b is not None else None
        )
        layer_type = "QuantLinearDequant"

    # Replace module in model hierarchy
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        parent_module = model.get_submodule(parent_name)
    else:
        parent_module = model
        child_name = name
    setattr(parent_module, child_name, new_mod)

    print(f"{i:03d} {name:60s} -> {layer_type}")

print("\nHybrid replacement complete!")
model.eval()

initial_mem = torch.cuda.memory_allocated() / 1024**3
print(f"Initial model memory usage: {initial_mem:.2f} GB")

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



# Increase batch size and sequence length for better GPU utilization
batch_size = 8  # Increased from 4
seq_len = 256   # Increased from 128

inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)

# Warmup with more iterations for better cache behavior
warmup_iters = 5
with torch.no_grad():
    for _ in range(warmup_iters):
        _ = model(inputs)

# Timed run with more iterations for better measurement
iters = 50  # Increased from 20
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
