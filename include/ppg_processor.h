// =============================================================================
// ppg_processor.h - Heart rate, SpO2 and respiratory rate from raw MAX30102 PPG
// (Arduino-independent; unit-tested on the host)
//
//  HR   : IR -> band-pass 0.5-4 Hz -> adaptive-threshold peak detector -> median IBI
//  SpO2 : ratio-of-ratios R = (AC_red/DC_red)/(AC_ir/DC_ir), SpO2 = 110 - 25 R
//         (Maxim's empirical line - calibrate per enclosure for clinical use)
//  RR   : IR baseline (respiratory-induced intensity variation), decimated 25 -> 5 Hz,
//         band-pass 0.1-0.5 Hz, hysteresis zero-crossing breath detector
//  Motion gate: while the wearer moves, samples are discarded and every estimator is
//         re-armed afterwards (filters need to settle before values become valid again).
// =============================================================================
#pragma once

#include <math.h>
#include <stdint.h>
#include "dsp_filters.h"

struct VitalsEstimate {
    float hr = NAN, spo2 = NAN, rr = NAN;
    bool hrValid = false, spo2Valid = false, rrValid = false;
    bool contact = false, gated = false;
};

class PpgProcessor {
public:
    explicit PpgProcessor(float fs = 25.0f, int rr_decimation = 5, uint32_t contact_min_ir = 30000);
    void reset();
    // Feed one sample. motion = true discards it and re-arms the estimators.
    void push(uint32_t red, uint32_t ir, bool motion);
    VitalsEstimate estimate() const { return est_; }

private:
    void rearm(float red, float ir);
    void updateHr(float x);
    void updateSpo2(float red_ac, float ir_ac, float red, float ir);
    void updateRr(float ir);

    const float fs_;
    const int dec_;
    const uint32_t contactMin_;

    dsp::BandpassFilter bpIr_, bpRed_, bpResp_;
    dsp::Ema dcIr_, dcRed_, acIr2_, acRed2_, spo2Smooth_, dcSlow_, respEnv_;

    uint32_t n_ = 0;               // samples since (re)arm
    bool armed_ = false;

    // HR beat detector
    float env_ = 0, prev1_ = 0, prev2_ = 0;
    uint32_t lastBeat_ = 0;
    bool haveBeat_ = false;
    static constexpr int kIbi = 10;
    float ibi_[kIbi] = {0};
    uint32_t ibiT_[kIbi] = {0};
    int ibiN_ = 0, ibiHead_ = 0;

    // RR breath detector (decimated domain)
    float decAcc_ = 0;
    int decCount_ = 0;
    uint32_t nDec_ = 0;
    bool respHigh_ = false;
    static constexpr int kBr = 12;
    uint32_t brT_[kBr] = {0};
    int brN_ = 0, brHead_ = 0;

    VitalsEstimate est_;
};
