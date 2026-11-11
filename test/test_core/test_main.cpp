// =============================================================================
// Host unit tests:  pio test -e native
// Covers the hardware-independent modules the ESP32 tasks are built on.
// =============================================================================
#include <unity.h>
#include <math.h>

#include "config.h"
#include "dsp_filters.h"
#include "fall_detector.h"
#include "mock_sensors.h"
#include "ppg_processor.h"

void setUp() {}
void tearDown() {}

// ---------------------------------------------------------------- DSP
void test_resp_bandpass_response() {
    dsp::RespiratoryBandpass bp(5.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.25f, 1.0f, bp.gainAt(0.25f));   // 15 br/min passes
    TEST_ASSERT_LESS_THAN_FLOAT(0.1f, bp.gainAt(0.02f));        // baseline drift rejected
    TEST_ASSERT_LESS_THAN_FLOAT(0.1f, bp.gainAt(1.5f));         // cardiac band rejected
}

void test_resp_bandpass_extracts_breathing() {
    // DC + breathing (0.3 Hz) + cardiac (1.4 Hz) + drift: measure each tone's output
    // amplitude with a single-bin DFT (phase lag of the IIR does not matter).
    const float fs = 5.0f, pi = 3.14159265f;
    dsp::RespiratoryBandpass bp(fs);
    bp.reset(100.0f);
    double br_c = 0, br_s = 0, ca_c = 0, ca_s = 0;
    int m = 0;
    for (int n = 0; n < 1000; ++n) {                            // 200 s
        const float t = n / fs;
        const float x = 100.0f + sinf(2 * pi * 0.3f * t) + 0.8f * sinf(2 * pi * 1.4f * t) + 0.01f * t;
        const float y = bp.process(x);
        if (n < 250) continue;                                  // let the filter settle
        br_c += y * cosf(2 * pi * 0.3f * t); br_s += y * sinf(2 * pi * 0.3f * t);
        ca_c += y * cosf(2 * pi * 1.4f * t); ca_s += y * sinf(2 * pi * 1.4f * t);
        ++m;
    }
    const float breath = 2.0f * sqrtf(br_c * br_c + br_s * br_s) / m;
    const float cardiac = 2.0f * sqrtf(ca_c * ca_c + ca_s * ca_s) / m / 0.8f;
    TEST_ASSERT_FLOAT_WITHIN(0.25f, 1.0f, breath);              // breathing passes (~0 dB)
    TEST_ASSERT_LESS_THAN_FLOAT(0.05f, cardiac);                // cardiac attenuated > 26 dB
}

void test_biquad_reset_has_no_transient() {
    dsp::Biquad lp = dsp::designLowpass(25.0f, 4.0f);
    lp.reset(120000.0f);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 120000.0f, lp.process(120000.0f));
}

void test_median() {
    const float v[] = {5, 1, 4, 2, 3};
    TEST_ASSERT_EQUAL_FLOAT(3.0f, dsp::median(v, 5));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, dsp::median(v, 4));   // {1,2,4,5}
    const float w[] = {4, 1, 3, 2};
    TEST_ASSERT_EQUAL_FLOAT(2.5f, dsp::median(w, 4));
}

// ---------------------------------------------------------------- vitals
static VitalsEstimate runPpg(PpgProcessor& p, sim::MockPpg& g, uint32_t from_ms, uint32_t to_ms, bool motion) {
    for (uint32_t t = from_ms; t < to_ms; t += 1000 / PPG_SAMPLE_HZ) {
        const PpgSample s = g.sample(t, motion);
        p.push(s.red, s.ir, motion);
    }
    return p.estimate();
}

void test_vitals_rest() {
    PpgProcessor p(PPG_SAMPLE_HZ, RR_DECIMATION, PPG_CONTACT_MIN_IR);
    sim::MockPpg g(PPG_SAMPLE_HZ);
    const VitalsEstimate e = runPpg(p, g, 0, 14000, false);     // scenario rest phase
    TEST_ASSERT_TRUE(e.contact);
    TEST_ASSERT_TRUE(e.hrValid);
    TEST_ASSERT_FLOAT_WITHIN(5.0f, 73.0f, e.hr);
    TEST_ASSERT_TRUE(e.spo2Valid);
    TEST_ASSERT_FLOAT_WITHIN(1.5f, 97.5f, e.spo2);
}

void test_vitals_hypoxia_and_rr() {
    PpgProcessor p(PPG_SAMPLE_HZ, RR_DECIMATION, PPG_CONTACT_MIN_IR);
    sim::MockPpg g(PPG_SAMPLE_HZ);
    const VitalsEstimate e = runPpg(p, g, 140000, 185000, false);   // hypoxia episode
    TEST_ASSERT_TRUE(e.spo2Valid);
    TEST_ASSERT_FLOAT_WITHIN(2.0f, 87.0f, e.spo2);
    TEST_ASSERT_FLOAT_WITHIN(6.0f, 122.0f, e.hr);
    TEST_ASSERT_TRUE(e.rrValid);
    TEST_ASSERT_FLOAT_WITHIN(3.0f, 25.0f, e.rr);
}

void test_vitals_motion_gate() {
    PpgProcessor p(PPG_SAMPLE_HZ, RR_DECIMATION, PPG_CONTACT_MIN_IR);
    sim::MockPpg g(PPG_SAMPLE_HZ);
    runPpg(p, g, 0, 12000, false);
    const VitalsEstimate e = runPpg(p, g, 15000, 20000, true);
    TEST_ASSERT_TRUE(e.gated);
    TEST_ASSERT_FALSE(e.hrValid);
    TEST_ASSERT_FALSE(e.spo2Valid);
}

void test_vitals_no_contact() {
    PpgProcessor p(PPG_SAMPLE_HZ, RR_DECIMATION, PPG_CONTACT_MIN_IR);
    for (int i = 0; i < 200; ++i) p.push(800, 1200, false);     // ambient light only
    TEST_ASSERT_FALSE(p.estimate().contact);
    TEST_ASSERT_FALSE(p.estimate().hrValid);
}

// ---------------------------------------------------------------- fall state machine
struct FallRun { int countdowns = 0, falls = 0, cancels = 0, inferences = 0; float firstCountdownS = -1; };

static FallRun runImu(uint32_t from_ms, uint32_t to_ms, uint32_t cancel_at_ms = 0) {
    sim::MockImu imu;
    FallDetector d;
    FallRun r;
    for (uint32_t t = from_ms; t <= to_ms; t += 1000 / MOTION_SAMPLE_HZ) {
        const bool cancel = cancel_at_ms && t >= cancel_at_ms && d.state() == FallState::COUNTDOWN;
        const FallDetector::Output o = d.update(imu.sample(t), cancel);
        if (o.prob >= 0) ++r.inferences;
        if (o.startCountdown) { ++r.countdowns; if (r.firstCountdownS < 0) r.firstCountdownS = t / 1000.0f; }
        r.falls += o.fallConfirmed;
        r.cancels += o.cancelled;
    }
    return r;
}

void test_fall_detected_and_alerted() {
    const FallRun r = runImu(50000, 80000);
    TEST_ASSERT_EQUAL(1, r.countdowns);
    TEST_ASSERT_EQUAL(1, r.falls);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 58.4f, r.firstCountdownS);  // impact 55.36 s + 0.5 s settle + 2.5 s still
}

void test_fall_cancelled_by_button() {
    const FallRun r = runImu(100000, 130000, 120000);
    TEST_ASSERT_EQUAL(1, r.countdowns);
    TEST_ASSERT_EQUAL(0, r.falls);
    TEST_ASSERT_EQUAL(1, r.cancels);
}

void test_jump_is_not_a_fall() {
    const FallRun r = runImu(30000, 45000);                    // jump at 35 s, keeps walking
    TEST_ASSERT_EQUAL(0, r.inferences);
    TEST_ASSERT_EQUAL(0, r.countdowns);
}

void test_table_slap_is_not_a_fall() {
    const FallRun r = runImu(126000, 140000);                  // 4 g spike without free fall
    TEST_ASSERT_EQUAL(0, r.inferences);
}

void test_impact_only_trigger_catches_fall_without_free_fall() {
    // A slide/faint: no free-fall valley, a 3 g impact, then lying still at a new orientation.
    FallDetector d;
    int inferences = 0, countdowns = 0;
    uint32_t t = 0;
    auto feed = [&](float ax, float ay, float az) {
        const FallDetector::Output o = d.update(ImuSample{ax, ay, az, 0.5f, -0.3f, 0.2f, t}, false);
        inferences += o.prob >= 0;
        countdowns += o.startCountdown;
        t += 20;
    };
    for (int i = 0; i < 150; ++i) feed(0.15f, -0.25f, 0.956f);           // standing, wrist down
    for (int i = 0; i < 25; ++i) feed(0.15f, -0.25f + 0.02f * i, 0.9f);  // slow slide, |a| ~ 0.9-1.1 g
    for (int i = 0; i < 4; ++i) feed(1.5f, 2.2f, 1.0f);                   // 2.8 g impact, no free fall
    for (int i = 0; i < 200; ++i) feed(0.92f, 0.30f, -0.25f);            // lying still on the side
    TEST_ASSERT_EQUAL(1, inferences);                                     // CNN/score was consulted
    TEST_ASSERT_EQUAL(1, countdowns);                                     // and the fallback score agrees
}

void test_walking_sets_motion_gate() {
    sim::MockImu imu;
    FallDetector d;
    bool active = false;
    for (uint32_t t = 20000; t <= 25000; t += 20) active = d.update(imu.sample(t), false).motionActive;
    TEST_ASSERT_TRUE(active);
    for (uint32_t t = 0; t <= 5000; t += 20) active = d.update(imu.sample(t + 240000), false).motionActive;
    TEST_ASSERT_FALSE(active);                                  // resting again in the next cycle
}

void test_window_alignment() {
    sim::MockImu imu;
    FallDetector d;
    for (uint32_t t = 54000; t <= 58500; t += 20) d.update(imu.sample(t), false);
    const float* w = d.window();
    int peak = 0;
    float best = 0;
    for (int i = 0; i < FALL_WINDOW_SAMPLES; ++i) {
        const float m = sqrtf(w[i * 6] * w[i * 6] + w[i * 6 + 1] * w[i * 6 + 1] + w[i * 6 + 2] * w[i * 6 + 2]);
        if (m > best) { best = m; peak = i; }
    }
    TEST_ASSERT_INT_WITHIN(3, FALL_IMPACT_INDEX, peak);         // impact where the CNN expects it
}

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_resp_bandpass_response);
    RUN_TEST(test_resp_bandpass_extracts_breathing);
    RUN_TEST(test_biquad_reset_has_no_transient);
    RUN_TEST(test_median);
    RUN_TEST(test_vitals_rest);
    RUN_TEST(test_vitals_hypoxia_and_rr);
    RUN_TEST(test_vitals_motion_gate);
    RUN_TEST(test_vitals_no_contact);
    RUN_TEST(test_fall_detected_and_alerted);
    RUN_TEST(test_fall_cancelled_by_button);
    RUN_TEST(test_jump_is_not_a_fall);
    RUN_TEST(test_table_slap_is_not_a_fall);
    RUN_TEST(test_impact_only_trigger_catches_fall_without_free_fall);
    RUN_TEST(test_walking_sets_motion_gate);
    RUN_TEST(test_window_alignment);
    return UNITY_END();
}
