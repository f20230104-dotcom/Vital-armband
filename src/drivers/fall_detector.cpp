// =============================================================================
// fall_detector.cpp - see fall_detector.h for the state diagram
// =============================================================================
#include "fall_detector.h"

#include <math.h>
#include <string.h>

namespace {
constexpr int kW = FALL_WINDOW_SAMPLES;
constexpr int kSnapshotDelay = FALL_WINDOW_SAMPLES - 1 - FALL_IMPACT_INDEX;   // samples after impact
inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
}

FallDetector::FallDetector() {
    memset(ring_, 0, sizeof(ring_));
    memset(win_, 0, sizeof(win_));
}

void FallDetector::enter(FallState s, uint32_t t, Output& o, const char* note) {
    state_ = s;
    tState_ = t;
    o.stateChanged = true;
    o.note = note;
}

void FallDetector::snapshotWindow() {
    // oldest sample is at head_ (the slot that will be overwritten next)
    for (int i = 0; i < kW; ++i) memcpy(&win_[i * 6], ring_[(head_ + i) % kW], 6 * sizeof(float));
    windowReady_ = true;
}

FallDetector::Output FallDetector::update(const ImuSample& s, bool cancelPressed) {
    Output o;
    float* r = ring_[head_];
    r[0] = s.ax; r[1] = s.ay; r[2] = s.az;
    r[3] = s.gx / FALL_GYRO_SCALE_DPS; r[4] = s.gy / FALL_GYRO_SCALE_DPS; r[5] = s.gz / FALL_GYRO_SCALE_DPS;
    head_ = (head_ + 1) % kW;
    if (filled_ < kW) ++filled_;

    const float mag = sqrtf(s.ax * s.ax + s.ay * s.ay + s.az * s.az);
    const float gmag = sqrtf(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz);

    // ---- motion gate for the PPG (independent of the fall state machine) ----------
    accStats_.push(mag);
    const bool active = accStats_.stddev() > MOTION_GATE_ACC_STD_G || gmag > MOTION_GATE_GYRO_DPS;
    if (active) lastMotion_ = s.t_ms ? s.t_ms : 1;
    motion_ = active || (lastMotion_ && s.t_ms - lastMotion_ < MOTION_GATE_HOLD_MS);
    o.motionActive = motion_;

    const uint32_t t = s.t_ms;
    switch (state_) {
        case FallState::IDLE:
            if (mag < FREEFALL_THRESHOLD_G) {
                if (ffCount_ == 0) tFfStart_ = t;
                if (++ffCount_ >= FREEFALL_MIN_SAMPLES) enter(FallState::FREE_FALL, t, o, "free fall detected");
            } else {
                ffCount_ = 0;
            }
            break;

        case FallState::FREE_FALL:
            if (mag > IMPACT_THRESHOLD_G) {
                tImpact_ = t;
                peakG_ = mag;
                ffMs_ = (float)(t - tFfStart_);
                samplesSinceImpact_ = 0;
                windowReady_ = false;
                enter(FallState::IMPACT, t, o, "impact after free fall");
            } else if (t - tFfStart_ > IMPACT_WINDOW_MS) {
                ffCount_ = 0;
                enter(FallState::IDLE, t, o, "free fall without impact (e.g. arm drop)");
            }
            break;

        case FallState::IMPACT:
            ++samplesSinceImpact_;
            if (mag > peakG_) peakG_ = mag;
            if (samplesSinceImpact_ == kSnapshotDelay) snapshotWindow();
            if (t - tImpact_ >= POST_IMPACT_SETTLE_MS) {
                stillViolations_ = stillSamples_ = 0;
                enter(FallState::POST_IMPACT, t, o, "checking for post-impact stillness");
            }
            break;

        case FallState::POST_IMPACT: {
            ++samplesSinceImpact_;
            if (samplesSinceImpact_ == kSnapshotDelay) snapshotWindow();
            ++stillSamples_;
            if (fabsf(mag - 1.0f) > STILLNESS_ACC_DEV_G || gmag > STILLNESS_GYRO_DPS) ++stillViolations_;
            const int allowed = stillSamples_ / 10 > 3 ? stillSamples_ / 10 : 3;   // tolerate ~10 % outliers
            if (stillViolations_ > allowed) {
                ffCount_ = 0;
                enter(FallState::IDLE, t, o, "movement after impact - wearer recovered");
            } else if (t - tImpact_ >= POST_IMPACT_SETTLE_MS + STILLNESS_MS && windowReady_) {
                enter(FallState::INFERENCE, t, o, "running classifier");
                prob_ = clf_ ? clf_(win_, ctx_) : 1.0f;
                o.prob = prob_;
                if (prob_ >= FALL_PROB_THRESHOLD) {
                    enter(FallState::COUNTDOWN, t, o, "fall confirmed by model - countdown");
                    o.startCountdown = true;
                } else {
                    ffCount_ = 0;
                    enter(FallState::IDLE, t, o, "model rejected (ADL)");
                }
            }
            break;
        }

        case FallState::INFERENCE:      // transient, handled inside POST_IMPACT
            break;

        case FallState::COUNTDOWN:
            if (cancelPressed) {
                o.cancelled = true;
                enter(FallState::COOLDOWN, t, o, "cancelled by wearer");
            } else if (t - tState_ >= (uint32_t)FALL_COUNTDOWN_S * 1000u) {
                o.fallConfirmed = true;
                enter(FallState::COOLDOWN, t, o, "no response - raising FALL alert");
            }
            break;

        case FallState::COOLDOWN:
            if (t - tState_ >= FALL_COOLDOWN_MS) {
                ffCount_ = 0;
                enter(FallState::IDLE, t, o, "re-armed");
            }
            break;
    }
    return o;
}

float FallDetector::heuristicScore(const float* w, void*) {
    auto mag = [&](int i) {
        const float* p = &w[i * 6];
        return sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    };
    const int c = FALL_IMPACT_INDEX;
    float minPre = 9, peak = 0;
    for (int i = c - 30; i < c; ++i) minPre = fminf(minPre, mag(i));
    for (int i = c - 5; i < c + 10; ++i) peak = fmaxf(peak, mag(i));
    // orientation before vs after (mean gravity vector)
    float pre[3] = {0}, post[3] = {0};
    for (int i = 0; i < c - 30; ++i) for (int k = 0; k < 3; ++k) pre[k] += w[i * 6 + k];
    dsp::RunningStats st(kW);
    for (int i = c + 25; i < kW; ++i) {
        for (int k = 0; k < 3; ++k) post[k] += w[i * 6 + k];
        st.push(mag(i));
    }
    const float np = sqrtf(pre[0] * pre[0] + pre[1] * pre[1] + pre[2] * pre[2]) + 1e-6f;
    const float nq = sqrtf(post[0] * post[0] + post[1] * post[1] + post[2] * post[2]) + 1e-6f;
    const float cosang = (pre[0] * post[0] + pre[1] * post[1] + pre[2] * post[2]) / (np * nq);
    const float angDeg = acosf(fmaxf(-1.f, fminf(1.f, cosang))) * 57.2958f;

    const float ff = clamp01((0.7f - minPre) / 0.4f);
    const float imp = clamp01((peak - 2.0f) / 2.0f);
    const float still = clamp01((0.15f - st.stddev()) / 0.10f);
    const float orient = clamp01(angDeg / 45.0f);
    return 0.25f * ff + 0.25f * imp + 0.25f * still + 0.25f * orient;
}
