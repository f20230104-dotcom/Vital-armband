// =============================================================================
// fall_detector.h - Threshold + ML fall-detection state machine (Arduino-independent)
//
//   IDLE --(|a| < 0.5 g for >= 60 ms)--> FREE_FALL
//   FREE_FALL --(|a| > 2.7 g within 1 s)--> IMPACT          (else back to IDLE)
//   IMPACT --(500 ms settle)--> POST_IMPACT
//   POST_IMPACT --(low motion for 2.5 s)--> INFERENCE        (movement -> IDLE: wearer recovered)
//   INFERENCE --(p(fall) >= threshold)--> COUNTDOWN          (else IDLE: model rejected)
//   COUNTDOWN --(button)--> COOLDOWN [FALL_CANCELLED]
//   COUNTDOWN --(15 s, no button)--> COOLDOWN [FALL]
//   COOLDOWN --(10 s)--> IDLE
//
// The 128-sample model window is snapshotted so the impact sits at FALL_IMPACT_INDEX,
// exactly like the training windows produced by ml_training/fetch_dataset.py.
// =============================================================================
#pragma once

#include <stdint.h>
#include "config.h"
#include "types.h"
#include "dsp_filters.h"

class FallDetector {
public:
    // Classifier callback: window is [FALL_WINDOW_SAMPLES][6] row-major in model units
    // (g for accel, dps / FALL_GYRO_SCALE_DPS for gyro). Returns p(fall) in [0, 1].
    using Classifier = float (*)(const float* window, void* ctx);

    struct Output {
        bool  stateChanged = false;
        bool  startCountdown = false;   // entered COUNTDOWN this sample -> start local alarm
        bool  fallConfirmed = false;    // countdown expired -> send FALL
        bool  cancelled = false;        // user cancelled -> send FALL_CANCELLED
        bool  motionActive = false;     // for the PPG motion gate
        float prob = -1.0f;             // last inference result (-1 = none this sample)
        const char* note = nullptr;     // human-readable reason for the transition
    };

    FallDetector();
    void setClassifier(Classifier fn, void* ctx) { clf_ = fn; ctx_ = ctx; }
    Output update(const ImuSample& s, bool cancelPressed);

    FallState state() const { return state_; }
    float lastProbability() const { return prob_; }
    float lastPeakG() const { return peakG_; }
    float lastFreeFallMs() const { return ffMs_; }
    const float* window() const { return win_; }

    // Rule-based fallback score used when no ML model is available.
    static float heuristicScore(const float* window, void* ctx = nullptr);

private:
    void enter(FallState s, uint32_t t, Output& o, const char* note);
    void snapshotWindow();

    FallState state_ = FallState::IDLE;
    Classifier clf_ = &FallDetector::heuristicScore;
    void* ctx_ = nullptr;

    // ring buffer of the last FALL_WINDOW_SAMPLES samples (model units)
    float ring_[FALL_WINDOW_SAMPLES][6];
    int head_ = 0, filled_ = 0;
    float win_[FALL_WINDOW_SAMPLES * 6];
    bool windowReady_ = false;

    uint32_t tState_ = 0, tImpact_ = 0, tFfStart_ = 0;
    int ffCount_ = 0, samplesSinceImpact_ = 0, stillViolations_ = 0, stillSamples_ = 0;
    float peakG_ = 0, prob_ = 0, ffMs_ = 0;

    // motion gate
    dsp::RunningStats accStats_{MOTION_SAMPLE_HZ};
    uint32_t lastMotion_ = 0;
    bool motion_ = false;
};
