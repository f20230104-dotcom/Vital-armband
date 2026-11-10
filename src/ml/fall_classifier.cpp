// =============================================================================
// fall_classifier.cpp - TensorFlow Lite Micro inference for the fall CNN
// =============================================================================
#include "fall_classifier.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "config.h"

#if FALL_USE_TFLM
#include "fall_model_data.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static_assert(FALL_MODEL_WINDOW_SAMPLES == FALL_WINDOW_SAMPLES, "model window != config.h FALL_WINDOW_SAMPLES");
static_assert(FALL_MODEL_IMPACT_INDEX == FALL_IMPACT_INDEX, "model impact index != config.h FALL_IMPACT_INDEX");
static_assert(FALL_MODEL_SAMPLE_HZ == MOTION_SAMPLE_HZ, "model sample rate != config.h MOTION_SAMPLE_HZ");

namespace {

constexpr int kInputLen = FALL_MODEL_WINDOW_SAMPLES * FALL_MODEL_CHANNELS;
alignas(16) uint8_t s_arena[FALL_MODEL_TENSOR_ARENA];

tflite::MicroInterpreter* s_interp = nullptr;
TfLiteTensor* s_in = nullptr;
TfLiteTensor* s_out = nullptr;
uint32_t s_lastUs = 0;
SemaphoreHandle_t s_lock = nullptr;

float invoke() {
    const uint32_t t0 = micros();
    if (s_interp->Invoke() != kTfLiteOk) return -1.0f;
    s_lastUs = micros() - t0;
    const float p = (s_out->data.int8[0] - s_out->params.zero_point) * s_out->params.scale;
    return p < 0 ? 0 : (p > 1 ? 1 : p);
}

}  // namespace

bool fallClassifierInit() {
    static tflite::MicroErrorReporter reporter;
    const tflite::Model* model = tflite::GetModel(g_fall_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[tflm] schema %lu != %d\n", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    // Only the 6 kernels this graph uses (see fall_model_data.h header comment) -> small binary
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddReshape();

    static tflite::MicroInterpreter interp(model, resolver, s_arena, sizeof(s_arena), &reporter);
    if (interp.AllocateTensors() != kTfLiteOk) {
        Serial.println("[tflm] AllocateTensors failed (increase FALL_MODEL_TENSOR_ARENA)");
        return false;
    }
    s_interp = &interp;
    s_in = interp.input(0);
    s_out = interp.output(0);
    if (s_in->type != kTfLiteInt8 || s_in->bytes != (size_t)kInputLen) {
        Serial.printf("[tflm] unexpected input tensor (type %d, %u bytes)\n", s_in->type, (unsigned)s_in->bytes);
        s_interp = nullptr;
        return false;
    }
    s_lock = xSemaphoreCreateMutex();
    Serial.printf("[tflm] model %u B, arena used %u / %u B, in scale %.5f zp %d\n", g_fall_model_tflite_len,
                  (unsigned)interp.arena_used_bytes(), (unsigned)sizeof(s_arena), s_in->params.scale,
                  (int)s_in->params.zero_point);
    return true;
}

bool fallClassifierReady() { return s_interp != nullptr; }

float fallClassifierPredict(const float* window, void*) {
    if (!s_interp) return -1.0f;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    const float inv = 1.0f / s_in->params.scale;
    const int zp = s_in->params.zero_point;
    int8_t* dst = s_in->data.int8;
    for (int i = 0; i < kInputLen; ++i) {
        int q = (int)lroundf(window[i] * inv) + zp;
        dst[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    const float p = invoke();
    xSemaphoreGive(s_lock);
    return p;
}

bool fallClassifierSelfTest(float* p_fall, float* p_adl) {
    if (!s_interp) return false;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    memcpy(s_in->data.int8, g_fall_selftest_fall, kInputLen);
    const float pf = invoke();
    memcpy(s_in->data.int8, g_fall_selftest_adl, kInputLen);
    const float pa = invoke();
    xSemaphoreGive(s_lock);
    if (p_fall) *p_fall = pf;
    if (p_adl) *p_adl = pa;
    return pf >= FALL_MODEL_THRESHOLD && pa < FALL_MODEL_THRESHOLD;
}

uint32_t fallClassifierLastMicros() { return s_lastUs; }

#else   // FALL_USE_TFLM == 0 : no TensorFlow dependency, motion_task uses the rule-based score
bool fallClassifierInit() { return false; }
bool fallClassifierReady() { return false; }
float fallClassifierPredict(const float*, void*) { return -1.0f; }
bool fallClassifierSelfTest(float*, float*) { return false; }
uint32_t fallClassifierLastMicros() { return 0; }
#endif
