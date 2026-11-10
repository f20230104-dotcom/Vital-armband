// =============================================================================
// ppg_processor.cpp - see ppg_processor.h for the signal chain
// =============================================================================
#include "ppg_processor.h"

#include <math.h>

namespace {
constexpr float kHrSettleS   = 1.5f;    // band-pass settling after (re)arm
constexpr float kHrWindowS   = 10.0f;   // IBIs used for the HR median
constexpr int   kHrMinBeats  = 4;
constexpr float kSpo2SettleS = 6.0f;
constexpr float kRrSettleS   = 8.0f;    // respiratory filters settle
constexpr float kRrValidS    = 20.0f;   // need a few breaths before reporting
constexpr float kRrWindowS   = 40.0f;
}

PpgProcessor::PpgProcessor(float fs, int rr_decimation, uint32_t contact_min_ir)
    : fs_(fs), dec_(rr_decimation), contactMin_(contact_min_ir) {
    bpIr_.configure(fs, 0.5f, 4.0f, 2);
    bpRed_.configure(fs, 0.5f, 4.0f, 2);
    const float fsDec = fs / rr_decimation;
    bpResp_.configure(fsDec, 0.10f, 0.50f, 2);          // the 0.1-0.5 Hz respiratory band
    dcIr_.configure(fs, 3.0f);
    dcRed_.configure(fs, 3.0f);
    acIr2_.configure(fs, 3.0f);
    acRed2_.configure(fs, 3.0f);
    spo2Smooth_.configure(fs, 3.0f);
    dcSlow_.configure(fsDec, 15.0f);
    respEnv_.configure(fsDec, 8.0f);
}

void PpgProcessor::reset() {
    armed_ = false;
    est_ = VitalsEstimate();
}

void PpgProcessor::rearm(float red, float ir) {
    bpIr_.reset(ir);
    bpRed_.reset(red);
    dcIr_.set(ir); dcRed_.set(red);
    acIr2_.reset(); acRed2_.reset(); spo2Smooth_.reset();
    n_ = 0;
    env_ = prev1_ = prev2_ = 0;
    haveBeat_ = false;
    ibiN_ = ibiHead_ = 0;
    decAcc_ = 0; decCount_ = 0; nDec_ = 0;
    dcSlow_.set(ir);
    bpResp_.reset(0.0f);
    respEnv_.reset();
    respHigh_ = false;
    brN_ = brHead_ = 0;
    est_.hrValid = est_.spo2Valid = est_.rrValid = false;
}

void PpgProcessor::push(uint32_t red, uint32_t ir, bool motion) {
    est_.gated = motion;
    if (motion) {                       // keep last values for display, but flag them stale
        armed_ = false;
        est_.hrValid = est_.spo2Valid = est_.rrValid = false;
        return;
    }
    est_.contact = ir >= contactMin_;
    if (!est_.contact) {
        armed_ = false;
        est_.hrValid = est_.spo2Valid = est_.rrValid = false;
        est_.hr = est_.spo2 = est_.rr = NAN;
        return;
    }
    const float r = (float)red, i = (float)ir;
    if (!armed_) { rearm(r, i); armed_ = true; }
    ++n_;
    const float irAc = bpIr_.process(i);
    const float redAc = bpRed_.process(r);
    updateHr(-irAc);                    // blood volume up -> more absorption -> IR dips: invert
    updateSpo2(redAc, irAc, r, i);
    updateRr(i);
}

void PpgProcessor::updateHr(float x) {
    const uint32_t t = (uint32_t)(n_ * 1000.0f / fs_);
    const float decay = expf(-0.6931472f / (2.0f * fs_));          // envelope half-life 2 s
    env_ *= decay;
    if (x > env_) env_ = x;

    if (n_ > kHrSettleS * fs_ && prev1_ > prev2_ && prev1_ >= x && prev1_ > 0.45f * env_ && prev1_ > 0) {
        const uint32_t tb = t - (uint32_t)(1000.0f / fs_);
        if (!haveBeat_) {
            haveBeat_ = true;
            lastBeat_ = tb;
        } else {
            const uint32_t ibi = tb - lastBeat_;
            if (ibi >= 300) {                                   // refractory: max 200 bpm
                if (ibi <= 1600) {                              // min 37 bpm, else a missed beat
                    ibi_[ibiHead_] = (float)ibi;
                    ibiT_[ibiHead_] = tb;
                    ibiHead_ = (ibiHead_ + 1) % kIbi;
                    if (ibiN_ < kIbi) ++ibiN_;
                }
                lastBeat_ = tb;
            }
        }
    }
    prev2_ = prev1_;
    prev1_ = x;

    float recent[kIbi];
    int m = 0;
    for (int k = 0; k < ibiN_; ++k)
        if (t - ibiT_[k] <= (uint32_t)(kHrWindowS * 1000)) recent[m++] = ibi_[k];
    if (m >= kHrMinBeats) {
        est_.hr = 60000.0f / dsp::median(recent, m);
        est_.hrValid = true;
    } else {
        est_.hrValid = false;
    }
}

void PpgProcessor::updateSpo2(float red_ac, float ir_ac, float red, float ir) {
    const float dcI = dcIr_.process(ir), dcR = dcRed_.process(red);
    const float aI = acIr2_.process(ir_ac * ir_ac), aR = acRed2_.process(red_ac * red_ac);
    if (n_ < 4 * fs_ || aI <= 0 || dcI <= 0 || dcR <= 0) return;
    const float R = (sqrtf(aR) / dcR) / (sqrtf(aI) / dcI);
    float s = 110.0f - 25.0f * R;
    s = s > 100.0f ? 100.0f : (s < 70.0f ? 70.0f : s);
    const float sm = spo2Smooth_.process(s);
    if (n_ >= kSpo2SettleS * fs_ && R > 0.3f && R < 1.6f) {
        est_.spo2 = sm;
        est_.spo2Valid = true;
    }
}

void PpgProcessor::updateRr(float ir) {
    decAcc_ += ir;
    if (++decCount_ < dec_) return;
    const float x = decAcc_ / dec_;
    decAcc_ = 0; decCount_ = 0;
    ++nDec_;
    const float fsDec = fs_ / dec_;
    const float dc = dcSlow_.process(x);
    const float y = bpResp_.process(x / dc - 1.0f);           // normalised baseline, 0.1-0.5 Hz
    const float env = respEnv_.process(fabsf(y));
    if (nDec_ < kRrSettleS * fsDec) return;

    const float thr = 0.35f * env;
    if (!respHigh_ && y > thr) {                               // upward crossing = one breath
        respHigh_ = true;
        brT_[brHead_] = nDec_;
        brHead_ = (brHead_ + 1) % kBr;
        if (brN_ < kBr) ++brN_;
    } else if (respHigh_ && y < -thr) {
        respHigh_ = false;
    }

    // intervals between consecutive breaths within the window
    uint32_t times[kBr];
    int m = 0;
    for (int k = 0; k < brN_; ++k) {
        const int idx = (brHead_ - brN_ + k + kBr) % kBr;      // chronological order
        if (nDec_ - brT_[idx] <= (uint32_t)(kRrWindowS * fsDec)) times[m++] = brT_[idx];
    }
    float iv[kBr];
    int ni = 0;
    for (int k = 1; k < m; ++k) iv[ni++] = (float)(times[k] - times[k - 1]);
    if (ni >= 3 && nDec_ >= kRrValidS * fsDec) {
        const float rr = 60.0f * fsDec / dsp::median(iv, ni);
        est_.rr = rr;
        est_.rrValid = rr >= 4.0f && rr <= 40.0f;
    } else {
        est_.rrValid = false;
    }
}
