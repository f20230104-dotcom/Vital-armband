// =============================================================================
// dsp_filters.h - Small, allocation-free DSP blocks (Arduino-independent)
// =============================================================================
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace dsp {

// Second-order IIR section, transposed direct form II (good numerical behaviour in float).
struct Biquad {
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float z1 = 0, z2 = 0;

    inline float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
    // Put the filter in the steady state it would reach for a constant input x
    // (avoids the huge start-up transient when x has a large DC offset, e.g. raw PPG).
    void reset(float x = 0.0f);
    float dcGain() const { return (b0 + b1 + b2) / (1.0f + a1 + a2); }
};

// RBJ audio-EQ-cookbook designs. q = 0.7071 gives a Butterworth section.
Biquad designLowpass(float fs, float fc, float q = 0.70710678f);
Biquad designHighpass(float fs, float fc, float q = 0.70710678f);
Biquad designBandpass(float fs, float f0, float q);   // 0 dB peak gain at f0

// Band-pass built from `sections` high-pass + `sections` low-pass Butterworth biquads
// (sections = 2 -> 4th-order skirts on each side).
class BandpassFilter {
public:
    static constexpr int kMaxSections = 3;
    BandpassFilter() = default;
    BandpassFilter(float fs, float f_low, float f_high, int sections = 2) { configure(fs, f_low, f_high, sections); }
    void configure(float fs, float f_low, float f_high, int sections = 2);
    float process(float x);
    void reset(float x = 0.0f);
    float gainAt(float f) const;          // magnitude response, for tests / docs

private:
    Biquad hp_[kMaxSections], lp_[kMaxSections];
    int n_ = 0;
    float fs_ = 1;
};

// Respiratory band: 0.1-0.5 Hz (6-30 breaths/min) used on the decimated PPG baseline.
class RespiratoryBandpass : public BandpassFilter {
public:
    explicit RespiratoryBandpass(float fs = 5.0f) : BandpassFilter(fs, 0.10f, 0.50f, 2) {}
};

// First-order exponential smoother with time constant tau (s).
class Ema {
public:
    Ema() = default;
    Ema(float fs, float tau_s) { configure(fs, tau_s); }
    void configure(float fs, float tau_s);
    inline float process(float x) {
        if (!init_) { y_ = x; init_ = true; } else { y_ += alpha_ * (x - y_); }
        return y_;
    }
    float value() const { return y_; }
    bool ready() const { return init_; }
    void reset() { init_ = false; y_ = 0; }
    void set(float v) { y_ = v; init_ = true; }
private:
    float alpha_ = 1, y_ = 0;
    bool init_ = false;
};

// Running mean / standard deviation over the last N samples (N <= kMax).
class RunningStats {
public:
    static constexpr int kMax = 128;
    explicit RunningStats(int n = 50) : n_(n > kMax ? kMax : (n < 1 ? 1 : n)) {}
    void push(float x);
    float mean() const { return count_ ? sum_ / count_ : 0.0f; }
    float stddev() const;
    int count() const { return count_; }
    void reset() { count_ = head_ = 0; sum_ = sum2_ = 0; }
private:
    float buf_[kMax] = {0};
    int n_, count_ = 0, head_ = 0;
    double sum_ = 0, sum2_ = 0;
};

// Median of up to 32 values (copies, does not modify input).
float median(const float* v, int n);

}  // namespace dsp
