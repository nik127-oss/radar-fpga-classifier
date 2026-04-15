#ifndef CNN_ACCEL_H
#define CNN_ACCEL_H

#include <stdint.h>

// Top-level function for Vivado HLS
// Interface: AXI-Lite (simple, no DMA needed for 64 bytes)
void cnn_accelerator(
    int8_t input_features[64],
    int *predicted_class,
    int *confidence
);

#endif
