"""
Run this AFTER Step3c_Calibrated_Export.py
Saves the calibrated weights + quant params for the PYNQ board.

Usage: python save_pynq_weights.py
"""
import numpy as np
import json
import re

# Read M1, M2, M3, ACT_MAX, REQUANT_SHIFT from weights.h
params = {}
with open('weights.h', 'r') as f:
    for line in f:
        m = re.match(r'#define\s+(\w+)\s+(\d+)', line)
        if m:
            params[m.group(1)] = int(m.group(2))

print("Quant params from weights.h:")
for k, v in params.items():
    print(f"  {k} = {v}")

# Save params as JSON for PYNQ
with open('quant_params.json', 'w') as f:
    json.dump(params, f)
print("\n✅ Saved: quant_params.json")

# Parse weight arrays from weights.h
def parse_array(filename, array_name):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find the array
    pattern = rf'const\s+int\d+_t\s+{array_name}\[(\d+)\]\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"  WARNING: Could not find {array_name}")
        return None
    
    size = int(match.group(1))
    values = [int(x.strip()) for x in match.group(2).split(',') if x.strip()]
    
    # Determine dtype from type name
    if 'int8_t' in content.split(array_name)[0].split('\n')[-1]:
        return np.array(values, dtype=np.int8)
    else:
        return np.array(values, dtype=np.int32)

arrays = {}
for name in ['conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias',
             'conv3_weight', 'conv3_bias', 'fc_weight', 'fc_bias']:
    arr = parse_array('weights.h', name)
    if arr is not None:
        arrays[name] = arr
        print(f"  {name}: {arr.shape}, dtype={arr.dtype}")

np.savez('weights_int8.npz', **arrays)
print("\n✅ Saved: weights_int8.npz")
print("\nUpload these to PYNQ:")
print("  - quant_params.json")
print("  - weights_int8.npz")
print("  - cnn_overlay.bit")
print("  - cnn_overlay.hwh")
print("  - dataset,npy")
print("  - X_test.npy, y_test.npy")
print("  - PYNQ_Live_Demo.ipynb")
