import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import math, time

from tqdm import tqdm
# --------------------
# Load model & tokenizer
# --------------------
model_name = "gpt2-large"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    dtype=torch.float16,
)
model.eval()

initial_mem = torch.cuda.memory_allocated() / 1024**3
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
print(f"Baseline Perplexity (GPT-2 Large): {ppl.item():.2f}")


batch_size = 8
seq_len = 256
inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)

# Warmup
warmup_iters = 5
with torch.no_grad():
    for _ in range(warmup_iters):
        _ = model(inputs)


# Timed run
iters = 50
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

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak GPU memory usage during run: {peak_mem:.2f} GB")
