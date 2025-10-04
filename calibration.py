import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

# --------------------
# Configuration
# --------------------
MODEL_NAME = "gpt2-large"
CALIBRATION_DATASET = "wikitext"
CALIBRATION_SUBSET = "wikitext-2-raw-v1"
NUM_CALIBRATION_SAMPLES = 30  # Number of text samples to use for calibration
MAX_LENGTH = 1024             # Max sequence length for the model
QUANTILE = 1.0             # The middle quantile range to keep (e.g., 0.999 = keep middle 99.9%)

# --------------------
# Main Calibration Logic
# --------------------
def main():
    print(f"Loading FP32 model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).cuda().eval()

    # --- 1. Find all the layers we want to calibrate ---
    layers_to_calibrate = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            layers_to_calibrate.append(name)
    
    print(f"Found {len(layers_to_calibrate)} layers to calibrate.")

    # --- 2. Set up hooks to capture activations ---
    # A dictionary to store the captured input activations for each layer
    captured_activations = defaultdict(list)

    def get_capture_hook(name):
        def hook_fn(module, input, output):
            # Detach and move to CPU to save GPU memory
            captured_activations[name].append(input[0].detach().cpu())
        return hook_fn

    hooks = []
    for name in layers_to_calibrate:
        module = model.get_submodule(name)
        hook = module.register_forward_hook(get_capture_hook(name))
        hooks.append(hook)

    # --- 3. Run calibration data through the model ---
    print(f"\nRunning {NUM_CALIBRATION_SAMPLES} samples from '{CALIBRATION_DATASET}' for calibration...")
    dataset = load_dataset(CALIBRATION_DATASET, CALIBRATION_SUBSET, split="train")
    
    with torch.no_grad():
        for i in tqdm(range(NUM_CALIBRATION_SAMPLES)):
            text = dataset[i]['text']
            if not text:
                continue
            
            inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to("cuda")
            model(inputs.input_ids)

    # --- 4. Remove hooks ---
    for hook in hooks:
        hook.remove()
    print("Calibration run complete. Hooks removed.")

    # --- 5. Process captured activations to find clipping ranges ---
    print("\nCalculating clipping ranges from captured activations...")
    calibration_ranges = {}
    
    # Calculate the min/max quantile values based on the desired range
    min_quantile = (1.0 - QUANTILE) / 2.0
    max_quantile = 1.0 - min_quantile

    for name, tensor_list in tqdm(captured_activations.items()):
        # ✅ FIX: Flatten each tensor individually first before concatenating
        # This handles the variable sequence lengths from different text samples.
        flattened_tensors = [t.view(-1, t.size(-1)) for t in tensor_list]
        
        # Now, concatenate the reshaped tensors along dim=0
        all_activations = torch.cat(flattened_tensors, dim=0)
        
        # The rest of the logic for calculating per-column quantiles is the same...
        q_min = torch.quantile(all_activations, min_quantile, dim=0)
        q_max = torch.quantile(all_activations, max_quantile, dim=0)
        
        calibration_ranges[name] = {'min': q_min.cpu(), 'max': q_max.cpu()}

    # --- 6. Save the calibration ranges to a file ---
    output_path = "calibration_ranges.pt"
    torch.save(calibration_ranges, output_path)
    
    print(f"\n✅ Calibration complete! Ranges saved to '{output_path}'")
    
    # Optional: Print a few examples
    print("\nExample ranges:")
    for i, (name, ranges) in enumerate(calibration_ranges.items()):
        if i >= 5: break
        print(f"  - {name}: min={ranges['min']}, max={ranges['max']}")


if __name__ == "__main__":
    main()