// =============================================================================
// fall_classifier.h - TensorFlow Lite Micro wrapper for the int8 fall CNN
// (model bytes + quantisation parameters come from fall_model_data.h)
// =============================================================================
#pragma once

#include <stdint.h>

bool     fallClassifierInit();                 // builds the interpreter, false on any error
bool     fallClassifierReady();
// window: [FALL_WINDOW_SAMPLES][6] float, model units. Returns p(fall) in [0, 1].
float    fallClassifierPredict(const float* window, void* ctx);
// Runs the two reference windows embedded in fall_model_data.h.
bool     fallClassifierSelfTest(float* p_fall, float* p_adl);
uint32_t fallClassifierLastMicros();
