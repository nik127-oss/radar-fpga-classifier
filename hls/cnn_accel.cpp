/*
 * cnn_accel.cpp — Calibrated requantization version
 * After each conv layer: output = clamp((acc * M) >> REQUANT_SHIFT, 0, ACT_MAX)
 * M values computed from real data calibration
 */
#include "cnn_accel.h"
#include "weights.h"
#include <stdint.h>

void cnn_accelerator(
    int8_t input_features[64],
    int *predicted_class,
    int *confidence
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=input_features bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=predicted_class bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=confidence bundle=ctrl

    int32_t conv1_out[8][32];
    int32_t conv2_out[16][16];
    int32_t conv3_out[16][4];
    long long fc_out[4];

    // Quantize input: [0,255] -> [0, ACT_MAX]
    int32_t q_input[64];
    for (int i = 0; i < 64; i++) {
        #pragma HLS PIPELINE II=1
        // Treat int8 as unsigned: -128..127 -> 0..255
        int val = (int)input_features[i];
        if (val < 0) val += 256;
        q_input[i] = (int32_t)((val * ACT_MAX) / 255);
    }

    // ============================================================
    // LAYER 1: Conv1d(1->8, k=5, pad=2) + ReLU + Requant + MaxPool(2)
    // ============================================================
    CONV1_OC: for (int oc = 0; oc < 8; oc++) {
        CONV1_POOL: for (int op = 0; op < 32; op++) {
            #pragma HLS PIPELINE II=2
            int32_t max_val = 0;

            CONV1_MP: for (int p = 0; p < 2; p++) {
                int pos = op * 2 + p;
                long long acc = (long long)conv1_bias[oc];

                CONV1_K: for (int k = 0; k < 5; k++) {
                    int idx = pos + k - 2;
                    if (idx >= 0 && idx < 64) {
                        acc += (long long)q_input[idx] * (long long)conv1_weight[oc * 5 + k];
                    }
                }
                // ReLU
                if (acc < 0) acc = 0;
                // Requantize
                int32_t val = (int32_t)((acc * M1) >> REQUANT_SHIFT);
                if (val > ACT_MAX) val = ACT_MAX;
                if (val > max_val) max_val = val;
            }
            conv1_out[oc][op] = max_val;
        }
    }

    // ============================================================
    // LAYER 2: Conv1d(8->16, k=3, pad=1) + ReLU + Requant + MaxPool(2)
    // ============================================================
    CONV2_OC: for (int oc = 0; oc < 16; oc++) {
        CONV2_POOL: for (int op = 0; op < 16; op++) {
            #pragma HLS PIPELINE II=4
            int32_t max_val = 0;

            CONV2_MP: for (int p = 0; p < 2; p++) {
                int pos = op * 2 + p;
                long long acc = (long long)conv2_bias[oc];

                CONV2_IC: for (int ic = 0; ic < 8; ic++) {
                    CONV2_K: for (int k = 0; k < 3; k++) {
                        int idx = pos + k - 1;
                        if (idx >= 0 && idx < 32) {
                            acc += (long long)conv1_out[ic][idx] * (long long)conv2_weight[(oc*8+ic)*3+k];
                        }
                    }
                }
                if (acc < 0) acc = 0;
                int32_t val = (int32_t)((acc * M2) >> REQUANT_SHIFT);
                if (val > ACT_MAX) val = ACT_MAX;
                if (val > max_val) max_val = val;
            }
            conv2_out[oc][op] = max_val;
        }
    }

    // ============================================================
    // LAYER 3: Conv1d(16->16, k=3, pad=1) + ReLU + Requant + AvgPool(4)
    // ============================================================
    CONV3_OC: for (int oc = 0; oc < 16; oc++) {
        CONV3_AVGPOOL: for (int op = 0; op < 4; op++) {
            #pragma HLS PIPELINE II=8
            long long pool_sum = 0;

            CONV3_PAVG: for (int p = 0; p < 4; p++) {
                int pos = op * 4 + p;
                long long acc = (long long)conv3_bias[oc];

                CONV3_IC: for (int ic = 0; ic < 16; ic++) {
                    CONV3_K: for (int k = 0; k < 3; k++) {
                        int idx = pos + k - 1;
                        if (idx >= 0 && idx < 16) {
                            acc += (long long)conv2_out[ic][idx] * (long long)conv3_weight[(oc*16+ic)*3+k];
                        }
                    }
                }
                if (acc < 0) acc = 0;
                int32_t val = (int32_t)((acc * M3) >> REQUANT_SHIFT);
                if (val > ACT_MAX) val = ACT_MAX;
                pool_sum += val;
            }
            conv3_out[oc][op] = (int32_t)(pool_sum / 4);
        }
    }

    // ============================================================
    // FC: Linear(64 -> 4) — no requantization, just argmax
    // ============================================================
    FC_CLASS: for (int c = 0; c < 4; c++) {
        #pragma HLS PIPELINE II=4
        long long sum = (long long)fc_bias[c];

        FC_J: for (int j = 0; j < 64; j++) {
            int oc = j / 4;
            int pos = j % 4;
            sum += (long long)conv3_out[oc][pos] * (long long)fc_weight[c * 64 + j];
        }
        fc_out[c] = sum;
    }

    // ARGMAX
    int best_class = 0;
    long long best_val = fc_out[0];
    ARGMAX: for (int c = 1; c < 4; c++) {
        if (fc_out[c] > best_val) {
            best_val = fc_out[c];
            best_class = c;
        }
    }

    *predicted_class = best_class;
    *confidence = (int)(best_val >> 16);
}
