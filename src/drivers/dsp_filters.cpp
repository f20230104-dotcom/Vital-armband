// =============================================================================
// dsp_filters.cpp - Biquad designs, the 0.1-0.5 Hz respiratory band-pass and helpers
// =============================================================================
#include "dsp_filters.h"

#include <math.h>
#include <string.h>

namespace dsp {

static constexpr float kPi = 3.14159265358979f;

void Biquad::reset(float x) {
    const float y = x * dcGain();
    z2 = b2 * x - a2 * y;
    z1 = y - b0 * x;
}

static Biquad normalise(float b0, float b1, float b2, float a0, float a1, float a2) {
    Biquad q;
    q.b0 = b0 / a0; q.b1 = b1 / a0; q.b2 = b2 / a0;
    q.a1 = a1 / a0; q.a2 = a2 / a0;
    return q;
}

Biquad designLowpass(float fs, float fc, float Q) {
    const float w0 = 2 * kPi * fc / fs, c = cosf(w0), alpha = sinf(w0) / (2 * Q);
    return normalise((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + alpha, -2 * c, 1 - alpha);
}

Biquad designHighpass(float fs, float fc, float Q) {
    const float w0 = 2 * kPi * fc / fs, c = cosf(w0), alpha = sinf(w0) / (2 * Q);
    return normalise((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + alpha, -2 * c, 1 - alpha);
}

Biquad designBandpass(float fs, float f0, float Q) {
    const float w0 = 2 * kPi * f0 / fs, c = cosf(w0), alpha = sinf(w0) / (2 * Q);
    return normalise(alpha, 0, -alpha, 1 + alpha, -2 * c, 1 - alpha);
}

// ---------------------------------------------------------------------------- band-pass
void BandpassFilter::configure(float fs, float f_low, float f_high, int sections) {
    n_ = sections < 1 ? 1 : (sections > kMaxSections ? kMaxSections : sections);
    fs_ = fs;
    for (int i = 0; i < n_; ++i) {
        hp_[i] = designHighpass(fs, f_low);
        lp_[i] = designLowpass(fs, f_high);
    }
}

float BandpassFilter::process(float x) {
    for (int i = 0; i < n_; ++i) x = hp_[i].process(x);
    for (int i = 0; i < n_; ++i) x = lp_[i].process(x);
    return x;
}

void BandpassFilter::reset(float x) {
    // first HP sees the DC level x; everything downstream sees 0
    for (int i = 0; i < n_; ++i) { hp_[i].reset(i == 0 ? x : 0.0f); lp_[i].reset(0.0f); }
}

static float biquadGain(const Biquad& q, float f, float fs) {
    const float w = 2 * kPi * f / fs;
    // H(e^jw) = (b0 + b1 e^-jw + b2 e^-2jw) / (1 + a1 e^-jw + a2 e^-2jw)
    const float nr = q.b0 + q.b1 * cosf(w) + q.b2 * cosf(2 * w);
    const float ni = -q.b1 * sinf(w) - q.b2 * sinf(2 * w);
    const float dr = 1 + q.a1 * cosf(w) + q.a2 * cosf(2 * w);
    const float di = -q.a1 * sinf(w) - q.a2 * sinf(2 * w);
    return sqrtf((nr * nr + ni * ni) / (dr * dr + di * di));
}

float BandpassFilter::gainAt(float f) const {
    float g = 1;
    for (int i = 0; i < n_; ++i) g *= biquadGain(hp_[i], f, fs_) * biquadGain(lp_[i], f, fs_);
    return g;
}

// ---------------------------------------------------------------------------- helpers
void Ema::configure(float fs, float tau_s) {
    alpha_ = 1.0f - expf(-1.0f / (fs * tau_s));
}

void RunningStats::push(float x) {
    if (count_ == n_) {
        const float old = buf_[head_];
        sum_ -= old; sum2_ -= (double)old * old;
    } else {
        ++count_;
    }
    buf_[head_] = x;
    sum_ += x; sum2_ += (double)x * x;
    head_ = (head_ + 1) % n_;
}

float RunningStats::stddev() const {
    if (count_ < 2) return 0.0f;
    const double m = sum_ / count_;
    const double v = sum2_ / count_ - m * m;
    return v > 0 ? (float)sqrt(v) : 0.0f;
}

float median(const float* v, int n) {
    if (n <= 0) return NAN;
    float tmp[32];
    if (n > 32) n = 32;
    memcpy(tmp, v, n * sizeof(float));
    for (int i = 1; i < n; ++i) {                    // insertion sort, n is tiny
        float k = tmp[i]; int j = i - 1;
        while (j >= 0 && tmp[j] > k) { tmp[j + 1] = tmp[j]; --j; }
        tmp[j + 1] = k;
    }
    return (n & 1) ? tmp[n / 2] : 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}

}  // namespace dsp
