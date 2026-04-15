"""
Run this on your PC to create a small demo file for PYNQ.
Extracts ~40 raw IQ samples (for spectrograms) + test features.

Input:  dataset,npy (~1GB), X_test.npy, y_test.npy
Output: pynq_demo_data.npz (~5-10 MB) — upload only this to PYNQ
"""
import numpy as np

print("Loading dataset...")
data = np.load('dataset,npy', allow_pickle=True)

def get_superclass(label):
    if hasattr(label, '__len__') and not isinstance(label, str):
        label = str(label[0]).strip()
    else:
        label = str(label).strip()
    if label.startswith('D'): return 0
    elif label in ['seagull','pigeon','raven','black-headed gull',
                    'seagull and black-headed gull','heron']: return 1
    elif label in ['human_walk','human_run']: return 2
    elif label == 'CR': return 3
    return -1

# Pick 10 samples per class (40 total) with raw IQ for spectrograms
demo_iq = []      # raw IQ vectors (complex, 1280 each)
demo_labels = []   # string labels
demo_classes = []  # superclass int

class_counts = {0:0, 1:0, 2:0, 3:0}
TARGET = 10

for row_idx in range(data.shape[0]):
    raw_label = data[row_idx, 0]
    label = str(raw_label[0]).strip() if hasattr(raw_label, '__len__') and not isinstance(raw_label, str) else str(raw_label).strip()
    sc = get_superclass(label)
    if sc == -1 or class_counts[sc] >= TARGET:
        continue
    
    iq = data[row_idx, 1][:, 0]  # first segment only
    demo_iq.append(iq)
    demo_labels.append(label)
    demo_classes.append(sc)
    class_counts[sc] += 1
    
    if all(v >= TARGET for v in class_counts.values()):
        break

demo_iq = np.array(demo_iq)        # (40, 1280) complex
demo_classes = np.array(demo_classes)  # (40,)

print(f"Demo IQ samples: {demo_iq.shape}")
for sc in range(4):
    n = (demo_classes == sc).sum()
    print(f"  Class {sc}: {n} samples")

# Load test features (already small)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Take a subset of test data (2000 samples max for speed on ARM)
N_SUB = min(2000, len(X_test))
idx = np.random.RandomState(42).permutation(len(X_test))[:N_SUB]
X_test_sub = X_test[idx]
y_test_sub = y_test[idx]

print(f"Test subset: {X_test_sub.shape}")

# Save everything in one file
np.savez_compressed('pynq_demo_data.npz',
    demo_iq_real=demo_iq.real.astype(np.float32),
    demo_iq_imag=demo_iq.imag.astype(np.float32),
    demo_labels=np.array(demo_labels),
    demo_classes=demo_classes,
    X_test=X_test_sub.astype(np.float32),
    y_test=y_test_sub,
)

import os
size_mb = os.path.getsize('pynq_demo_data.npz') / 1e6
print(f"\n✅ Saved: pynq_demo_data.npz ({size_mb:.1f} MB)")
print("\nUpload to PYNQ:")
print("  - pynq_demo_data.npz")
print("  - weights_int8.npz")
print("  - quant_params.json")
print("  - cnn_overlay.bit + cnn_overlay.hwh")
print("  - PYNQ_Live_Demo.ipynb")
print("\nDO NOT upload dataset,npy or X_test.npy/y_test.npy")
