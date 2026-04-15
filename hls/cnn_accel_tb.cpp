/*
 * cnn_accel_tb.cpp - Testbench for CNN accelerator
 */
#include <stdio.h>
#include <stdlib.h>
#include "cnn_accel.h"

int main() {
    int8_t test_input[64];
    int predicted_class;
    int confidence;

    // Fill with pseudo-random test data
    for (int i = 0; i < 64; i++) {
        test_input[i] = (int8_t)((i * 17 + 31) % 255 - 128);
    }

    printf("Running CNN accelerator...\n");
    cnn_accelerator(test_input, &predicted_class, &confidence);

    printf("Predicted class: %d\n", predicted_class);
    printf("Confidence: %d\n", confidence);

    // Sanity check
    if (predicted_class >= 0 && predicted_class <= 3) {
        printf("TEST PASSED\n");
        return 0;
    } else {
        printf("TEST FAILED: invalid class %d\n", predicted_class);
        return 1;
    }
}
