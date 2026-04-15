"""
Step 3c: Calibration-based fixed-point quantization.

Instead of computing bias scales mathematically, this:
1. Runs real data through the FLOAT model
2. Measures actual min/max of each layer's output
3. Computes exact scale factors to map float ranges to int ranges
4. Exports weights + scales that produce correct results in integer

This is how TFLite / CMSIS-NN / real production quantization works.
"""

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Load model and fuse BN
# ============================================================
class TinyCNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(16 * 4, num_classes)
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# Try QAT model first, fall back to original
try:
    model = TinyCNN1D()
    model.load_state_dict(torch.load('best_model_qat.pth', map_location='cpu'))
    print("Loaded QAT model")
except:
    model = TinyCNN1D()
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    print("Loaded original model")
model.eval()

def fuse_conv_bn(conv, bn):
    w = conv.weight.data.clone()
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
    gamma, beta = bn.weight.data, bn.bias.data
    mean, var, eps = bn.running_mean, bn.running_var, bn.eps
    scale = gamma / torch.sqrt(var + eps)
    return (w * scale.view(-1,1,1)).numpy(), ((b - mean) * scale + beta).numpy()

c1w_f, c1b_f = fuse_conv_bn(model.conv1, model.bn1)
c2w_f, c2b_f = fuse_conv_bn(model.conv2, model.bn2)
c3w_f, c3b_f = fuse_conv_bn(model.conv3, model.bn3)
fcw_f = model.fc.weight.data.numpy()
fcb_f = model.fc.bias.data.numpy()

# ============================================================
# Step 1: Run calibration data through FLOAT model layer by layer
#         to measure actual activation ranges
# ============================================================
X_train = np.load('X_train.npy')
N_CAL = min(2000, len(X_train))  # calibration samples
cal_data = X_train[:N_CAL].astype(np.float64) / 255.0  # [0, 1]

print(f"\nCalibrating with {N_CAL} samples...")

# Track min/max at each layer output
layer1_max = 0.0
layer2_max = 0.0
layer3_max = 0.0

c1w = c1w_f.reshape(8, 1, 5)
c2w = c2w_f.reshape(16, 8, 3)
c3w = c3w_f.reshape(16, 16, 3)

for idx in range(N_CAL):
    x = cal_data[idx].reshape(1, 1, 64)
    
    # Layer 1: Conv + ReLU + MaxPool
    out1 = np.zeros((1, 8, 64))
    for oc in range(8):
        for i in range(64):
            s = c1b_f[oc]
            for k in range(5):
                ii = i + k - 2
                if 0 <= ii < 64: s += x[0,0,ii] * c1w[oc,0,k]
            out1[0,oc,i] = max(0, s)
    p1 = np.zeros((1, 8, 32))
    for oc in range(8):
        for i in range(32): p1[0,oc,i] = max(out1[0,oc,2*i], out1[0,oc,2*i+1])
    layer1_max = max(layer1_max, p1.max())
    
    # Layer 2
    out2 = np.zeros((1, 16, 32))
    for oc in range(16):
        for i in range(32):
            s = c2b_f[oc]
            for ic in range(8):
                for k in range(3):
                    ii = i+k-1
                    if 0 <= ii < 32: s += p1[0,ic,ii] * c2w[oc,ic,k]
            out2[0,oc,i] = max(0, s)
    p2 = np.zeros((1, 16, 16))
    for oc in range(16):
        for i in range(16): p2[0,oc,i] = max(out2[0,oc,2*i], out2[0,oc,2*i+1])
    layer2_max = max(layer2_max, p2.max())
    
    # Layer 3
    out3 = np.zeros((1, 16, 16))
    for oc in range(16):
        for i in range(16):
            s = c3b_f[oc]
            for ic in range(16):
                for k in range(3):
                    ii = i+k-1
                    if 0 <= ii < 16: s += p2[0,ic,ii] * c3w[oc,ic,k]
            out3[0,oc,i] = max(0, s)
    p3 = np.zeros((1, 16, 4))
    for oc in range(16):
        for i in range(4): p3[0,oc,i] = np.mean(out3[0,oc,i*4:(i+1)*4])
    layer3_max = max(layer3_max, p3.max())
    
    if (idx+1) % 500 == 0:
        print(f"  Calibrated {idx+1}/{N_CAL}")

print(f"\nCalibrated activation ranges:")
print(f"  Input:  [0, 1]")
print(f"  Layer1: [0, {layer1_max:.4f}]")
print(f"  Layer2: [0, {layer2_max:.4f}]")
print(f"  Layer3: [0, {layer3_max:.4f}]")

# ============================================================
# Step 2: Compute quantization parameters
#
# For each layer:
#   float_output = sum(float_input * float_weight) + float_bias
#   
# We quantize activations to [0, 32767] (int16 unsigned after ReLU)
# and weights to [-127, 127] (int8 signed)
#
# Scales:
#   input_scale  = input_max / 32767
#   output_scale = output_max / 32767
#   weight_scale = weight_max / 127
#
# The accumulator: acc = sum(q_input * q_weight) + q_bias
#   has scale = input_scale * weight_scale
#
# To convert to output quantization:
#   q_output = acc * (input_scale * weight_scale / output_scale)
#   q_output = acc * M     where M = (input_scale * weight_scale / output_scale)
#
# We approximate M as (multiplier >> shift) for integer-only math:
#   multiplier = round(M * 2^shift)
# ============================================================

ACT_BITS = 15  # Use 15-bit activations (0 to 32767) for headroom
ACT_MAX = (1 << ACT_BITS) - 1  # 32767

# Activation scales (float value per integer unit)
s_input = 1.0 / ACT_MAX        # input range [0,1] mapped to [0, ACT_MAX]
s_act1 = layer1_max / ACT_MAX
s_act2 = layer2_max / ACT_MAX
s_act3 = layer3_max / ACT_MAX

# Weight quantization
def q_weight(w_float):
    flat = w_float.flatten()
    abs_max = max(abs(flat.min()), abs(flat.max()), 1e-10)
    scale = abs_max / 127.0
    q = np.clip(np.round(flat / scale), -128, 127).astype(np.int8)
    return q, scale

c1w_q, s_w1 = q_weight(c1w_f)
c2w_q, s_w2 = q_weight(c2w_f)
c3w_q, s_w3 = q_weight(c3w_f)
fcw_q, s_wf = q_weight(fcw_f)

# Requantization multipliers:
# After accumulation, acc has scale = s_input_act * s_weight
# We need to convert to output activation scale: s_output_act
# M = (s_input_act * s_weight) / s_output_act

RSHIFT = 15  # Fixed right-shift for all requantization

M1_float = (s_input * s_w1) / s_act1
M1 = int(round(M1_float * (1 << RSHIFT)))

M2_float = (s_act1 * s_w2) / s_act2
M2 = int(round(M2_float * (1 << RSHIFT)))

M3_float = (s_act2 * s_w3) / s_act3
M3 = int(round(M3_float * (1 << RSHIFT)))

# FC doesn't need requantization — just argmax
# But we need the accumulator scale for bias
s_fc_acc = s_act3 * s_wf

print(f"\nRequantization multipliers (right-shift by {RSHIFT}):")
print(f"  M1 = {M1} (float: {M1_float:.6f})")
print(f"  M2 = {M2} (float: {M2_float:.6f})")
print(f"  M3 = {M3} (float: {M3_float:.6f})")

# Bias quantization — bias must be in accumulator scale
# acc = sum(q_input * q_weight) + q_bias
# acc has implicit scale = s_input_act * s_weight
# So q_bias = float_bias / (s_input_act * s_weight)

c1b_q = np.round(c1b_f.flatten() / (s_input * s_w1)).astype(np.int32)
c2b_q = np.round(c2b_f.flatten() / (s_act1 * s_w2)).astype(np.int32)
c3b_q = np.round(c3b_f.flatten() / (s_act2 * s_w3)).astype(np.int32)
fcb_q = np.round(fcb_f.flatten() / (s_act3 * s_wf)).astype(np.int32)

print(f"\nBias ranges:")
print(f"  c1b: [{c1b_q.min()}, {c1b_q.max()}]")
print(f"  c2b: [{c2b_q.min()}, {c2b_q.max()}]")
print(f"  c3b: [{c3b_q.min()}, {c3b_q.max()}]")
print(f"  fcb: [{fcb_q.min()}, {fcb_q.max()}]")

# ============================================================
# Step 3: Integer forward pass with calibrated requantization
# ============================================================
def forward_calibrated(x_input):
    """
    Fixed-point forward pass using calibrated scales.
    Input: int8 features (will be treated as unsigned 0-255, same as HLS)
    """
    # Convert int8 to unsigned [0, 255] — same as HLS: if (val < 0) val += 256
    raw = [int(v) for v in x_input]
    unsigned = [v + 256 if v < 0 else v for v in raw]
    # Quantize to [0, ACT_MAX]
    inp = [min(ACT_MAX, max(0, (v * ACT_MAX) // 255)) for v in unsigned]
    
    # Conv1 + ReLU + MaxPool + requantize
    conv1_out = [[0]*32 for _ in range(8)]
    for oc in range(8):
        for op in range(32):
            mv = -(1 << 60)
            for p in range(2):
                pos = op * 2 + p
                acc = int(c1b_q[oc])
                for k in range(5):
                    idx = pos + k - 2
                    if 0 <= idx < 64:
                        acc += inp[idx] * int(c1w_q[oc * 5 + k])
                # ReLU
                if acc < 0: acc = 0
                # Requantize: multiply by M1, right-shift by RSHIFT
                acc = (acc * M1) >> RSHIFT
                # Clamp to activation range
                if acc > ACT_MAX: acc = ACT_MAX
                if acc > mv: mv = acc
            conv1_out[oc][op] = mv
    
    # Conv2 + ReLU + MaxPool + requantize
    conv2_out = [[0]*16 for _ in range(16)]
    for oc in range(16):
        for op in range(16):
            mv = -(1 << 60)
            for p in range(2):
                pos = op * 2 + p
                acc = int(c2b_q[oc])
                for ic in range(8):
                    for k in range(3):
                        idx = pos + k - 1
                        if 0 <= idx < 32:
                            acc += conv1_out[ic][idx] * int(c2w_q[(oc*8+ic)*3+k])
                if acc < 0: acc = 0
                acc = (acc * M2) >> RSHIFT
                if acc > ACT_MAX: acc = ACT_MAX
                if acc > mv: mv = acc
            conv2_out[oc][op] = mv
    
    # Conv3 + ReLU + AvgPool + requantize
    conv3_out = [[0]*4 for _ in range(16)]
    for oc in range(16):
        for op in range(4):
            pool_sum = 0
            for p in range(4):
                pos = op * 4 + p
                acc = int(c3b_q[oc])
                for ic in range(16):
                    for k in range(3):
                        idx = pos + k - 1
                        if 0 <= idx < 16:
                            acc += conv2_out[ic][idx] * int(c3w_q[(oc*16+ic)*3+k])
                if acc < 0: acc = 0
                acc = (acc * M3) >> RSHIFT
                if acc > ACT_MAX: acc = ACT_MAX
                pool_sum += acc
            conv3_out[oc][op] = pool_sum // 4
    
    # FC — no requantization, just argmax
    fc_out = [0] * 4
    for c in range(4):
        s = int(fcb_q[c])
        for j in range(64):
            s += conv3_out[j//4][j%4] * int(fcw_q[c*64+j])
        fc_out[c] = s
    
    return fc_out.index(max(fc_out))

# ============================================================
# Verify
# ============================================================
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
CLASS_NAMES = ['Drone', 'Bird', 'Human', 'CR']

# Also compute float accuracy for comparison
def forward_float(x_input):
    x = x_input.astype(np.float64) / 255.0
    x = x.reshape(1,1,64)
    c1 = c1w_f.reshape(8,1,5)
    c2 = c2w_f.reshape(16,8,3)
    c3 = c3w_f.reshape(16,16,3)
    
    o = np.zeros((1,8,64))
    for oc in range(8):
        for i in range(64):
            s=c1b_f[oc]
            for k in range(5):
                ii=i+k-2
                if 0<=ii<64: s+=x[0,0,ii]*c1[oc,0,k]
            o[0,oc,i]=max(0,s)
    p=np.zeros((1,8,32))
    for oc in range(8):
        for i in range(32): p[0,oc,i]=max(o[0,oc,2*i],o[0,oc,2*i+1])
    
    o2=np.zeros((1,16,32))
    for oc in range(16):
        for i in range(32):
            s=c2b_f[oc]
            for ic in range(8):
                for k in range(3):
                    ii=i+k-1
                    if 0<=ii<32: s+=p[0,ic,ii]*c2[oc,ic,k]
            o2[0,oc,i]=max(0,s)
    p2=np.zeros((1,16,16))
    for oc in range(16):
        for i in range(16): p2[0,oc,i]=max(o2[0,oc,2*i],o2[0,oc,2*i+1])
    
    o3=np.zeros((1,16,16))
    for oc in range(16):
        for i in range(16):
            s=c3b_f[oc]
            for ic in range(16):
                for k in range(3):
                    ii=i+k-1
                    if 0<=ii<16: s+=p2[0,ic,ii]*c3[oc,ic,k]
            o3[0,oc,i]=max(0,s)
    p3=np.zeros((1,16,4))
    for oc in range(16):
        for i in range(4): p3[0,oc,i]=np.mean(o3[0,oc,i*4:(i+1)*4])
    
    flat=p3.flatten()
    logits=fcb_f.copy()
    for c in range(4):
        for j in range(64): logits[c]+=flat[j]*fcw_f[c,j]
    return int(np.argmax(logits))

N_V = min(500, len(X_test))
float_ok, fixed_ok, match = 0, 0, 0
print(f"\nVerifying on {N_V} test samples...")
for i in range(N_V):
    feat = X_test[i].astype(np.uint8).astype(np.int8)  # [0,255] -> int8 (wraps >127)
    fp = forward_float(X_test[i])  # float uses original values
    ip = forward_calibrated(feat)  # fixed-point goes through int8->unsigned
    if fp == y_test[i]: float_ok += 1
    if ip == y_test[i]: fixed_ok += 1
    if fp == ip: match += 1
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{N_V}: float={100*float_ok/(i+1):.1f}% fixed={100*fixed_ok/(i+1):.1f}% match={100*match/(i+1):.1f}%")

print(f"\n{'='*50}")
print(f"Float accuracy:      {100*float_ok/N_V:.1f}%")
print(f"Fixed-point accuracy: {100*fixed_ok/N_V:.1f}%")
print(f"Float↔Fixed match:   {100*match/N_V:.1f}%")
print(f"{'='*50}")

# ============================================================
# Export weights.h
# ============================================================
def w8(f, name, data, cmt=""):
    if cmt: f.write(f"// {cmt}\n")
    f.write(f"const int8_t {name}[{len(data)}] = {{\n  ")
    for i,v in enumerate(data):
        f.write(f"{int(v)}")
        if i<len(data)-1: f.write(", ")
        if (i+1)%20==0 and i<len(data)-1: f.write("\n  ")
    f.write("\n};\n\n")
def w32(f, name, data, cmt=""):
    if cmt: f.write(f"// {cmt}\n")
    f.write(f"const int32_t {name}[{len(data)}] = {{\n  ")
    for i,v in enumerate(data):
        f.write(f"{int(v)}")
        if i<len(data)-1: f.write(", ")
        if (i+1)%12==0 and i<len(data)-1: f.write("\n  ")
    f.write("\n};\n\n")

with open('weights.h', 'w') as f:
    f.write("// Calibration-based fixed-point weights\n")
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n#include <stdint.h>\n\n")
    f.write(f"#define ACT_MAX {ACT_MAX}\n")
    f.write(f"#define REQUANT_SHIFT {RSHIFT}\n")
    f.write(f"#define M1 {M1}\n")
    f.write(f"#define M2 {M2}\n")
    f.write(f"#define M3 {M3}\n\n")
    w8(f, "conv1_weight", c1w_q)
    w32(f, "conv1_bias", c1b_q)
    w8(f, "conv2_weight", c2w_q)
    w32(f, "conv2_bias", c2b_q)
    w8(f, "conv3_weight", c3w_q)
    w32(f, "conv3_bias", c3b_q)
    w8(f, "fc_weight", fcw_q)
    w32(f, "fc_bias", fcb_q)
    f.write("#endif\n")
print("\n✅ weights.h exported")

# ============================================================
# Export testbench
# ============================================================
sel = []
for cls in range(4): sel.extend(np.where(y_test == cls)[0][:5])
tb_f, tb_t, tb_r = [], [], []
for idx in sel:
    feat_int8 = X_test[idx].astype(np.uint8).astype(np.int8)
    pred = forward_calibrated(feat_int8)
    tb_f.append(feat_int8)
    tb_t.append(int(y_test[idx]))
    tb_r.append(pred)
    m = "✓" if pred==y_test[idx] else "✗"
    print(f"  TB[{idx}] true={CLASS_NAMES[y_test[idx]]:5s} pred={CLASS_NAMES[pred]:5s} {m}")

with open('cnn_accel_tb.cpp', 'w') as f:
    f.write('#include <stdio.h>\n#include "cnn_accel.h"\n\n')
    f.write(f"#define N {len(tb_f)}\n")
    f.write("const int8_t td[N][64]={\n")
    for i,ft in enumerate(tb_f):
        f.write("{"+",".join(str(int(v)) for v in ft)+"}"+ ("," if i<len(tb_f)-1 else "")+"\n")
    f.write("};\n")
    f.write(f"const int tl[N]={{{','.join(str(v) for v in tb_t)}}};\n")
    f.write(f"const int rp[N]={{{','.join(str(v) for v in tb_r)}}};\n")
    f.write('const char* cn[4]={"Drone","Bird","Human","CR"};\n')
    f.write("""int main(){
    int8_t in[64];int pc,cf,ok=0,m=0;
    printf("=== Calibrated Fixed-Point Validation ===\\n");
    for(int s=0;s<N;s++){
        for(int i=0;i<64;i++)in[i]=td[s][i];
        cnn_accelerator(in,&pc,&cf);
        int t=(pc==tl[s]),r=(pc==rp[s]);
        if(t)ok++;if(r)m++;
        printf("%2d|%-6s|%-6s|%-6s|%s|%s\\n",s,cn[tl[s]],cn[rp[s]],cn[pc],t?"OK":"NO",r?"MATCH":"DIFF");
    }
    printf("Acc:%d/%d(%.1f%%) Match:%d/%d\\n",ok,N,100.0*ok/N,m,N);
    if(m>=N-1){printf("PASSED\\n");return 0;}
    else{printf("FAILED\\n");return 1;}
}
""")
tb_acc=100*sum(1 for p,t in zip(tb_r,tb_t) if p==t)/len(tb_t)
print(f"\n✅ cnn_accel_tb.cpp (ref acc: {tb_acc:.1f}%)")
