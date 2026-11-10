// =============================================================================
// mock_sensors.h - Simulated hardware for bench testing without sensors.
//
// Header-only and Arduino-free so the same generators run on the ESP32
// (SIMULATION_MODE) and in host unit tests (pio test -e native).
//
// A deterministic 240 s scenario is replayed in a loop:
//
//   t [s]   IMU (wrist)                       physiology / buttons
//   0-15    rest (sitting)                    HR 72, SpO2 97.5, RR 15
//   15-35   walking (arm swing)  -> PPG motion-gated
//   35      JUMP  (free fall + 3.8 g landing, keeps walking)  -> must NOT alert
//   40-55   rest
//   55      FALL forward -> lying still        HR climbs to ~105 (stress response)
//                         -> countdown expires -> FALL alert
//   95-110  walking
//   110     FALL lateral -> lying              button A short-press at 120 -> FALL_CANCELLED
//   125-140 gestures, SLAP on table at 132 (4 g spike, no free fall) -> must NOT alert
//   145-195 rest, hypoxia episode              SpO2 87 %, HR 122, RR 25 -> VITALS_ABNORMAL
//   225     button A long-press                -> SOS
//   Medication: reminders every SIM_MED_INTERVAL_S; even ones confirmed with button B
//   after 20 s, odd ones ignored -> MED_MISSED.
// =============================================================================
#pragma once

#include <math.h>
#include <stdint.h>
#include "types.h"

namespace sim {

constexpr float kCycleS = 240.0f;
constexpr float kPi = 3.14159265f;

// ---- tiny deterministic PRNG ----------------------------------------------------
class Rng {
public:
    explicit Rng(uint32_t seed = 0x1234567u) : s_(seed ? seed : 1u) {}
    uint32_t next() { s_ ^= s_ << 13; s_ ^= s_ >> 17; s_ ^= s_ << 5; return s_; }
    float uniform() { return (next() >> 8) * (1.0f / 16777216.0f); }
    float gauss(float sd = 1.0f) {
        float u1 = uniform() + 1e-7f, u2 = uniform();
        return sd * sqrtf(-2.0f * logf(u1)) * cosf(2 * kPi * u2);
    }
private:
    uint32_t s_;
};

struct Vec3 { float x, y, z; };
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator*(Vec3 a, float k) { return {a.x * k, a.y * k, a.z * k}; }
inline float norm(Vec3 a) { return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z); }
inline Vec3 unit(Vec3 a) { float n = norm(a); return n > 1e-6f ? a * (1.0f / n) : Vec3{0, 0, 1}; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// Rodrigues rotation of v around unit axis k by angle th
inline Vec3 rotate(Vec3 v, Vec3 k, float th) {
    float c = cosf(th), s = sinf(th);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1 - c));
}

// ---- scenario --------------------------------------------------------------------
enum class ImuMode : uint8_t { REST, WALK, GESTURE, LYING };

struct Physiology { float hr, spo2, rr; };

class Scenario {
public:
    static float cycleTime(uint32_t t_ms) { return fmodf(t_ms / 1000.0f, kCycleS); }

    static ImuMode mode(float ct) {
        if (ct < 15) return ImuMode::REST;
        if (ct < 40) return ImuMode::WALK;
        if (ct < 55) return ImuMode::REST;
        if (ct < 95) return ImuMode::LYING;
        if (ct < 110) return ImuMode::WALK;
        if (ct < 125) return ImuMode::LYING;
        if (ct < 140) return ImuMode::GESTURE;
        return ImuMode::REST;
    }

    static Physiology physiology(float ct) {
        Physiology p{72.0f + 2.0f * sinf(ct / 20.0f), 97.5f, 15.0f};
        if ((ct >= 15 && ct < 40) || (ct >= 95 && ct < 110)) p.hr = 95.0f;          // walking
        if (ct >= 55 && ct < 115) p.hr += 33.0f * expf(-(ct - 55) / 40.0f);       // post-fall stress
        if (ct >= 145 && ct < 195) {                                              // hypoxia episode
            float ramp = fminf(1.0f, (ct - 145) / 6.0f) * fminf(1.0f, (195 - ct) / 4.0f);
            p.spo2 -= 10.5f * ramp; p.hr += 50.0f * ramp; p.rr += 10.0f * ramp;
        }
        return p;
    }

    // Simulated button activity between two timestamps (edge semantics).
    // Returns bitmask of EVT_BTN_* that occurred in (prev_ms, now_ms].
    static uint32_t buttons(uint32_t prev_ms, uint32_t now_ms, uint32_t med_first_s, uint32_t med_interval_s) {
        uint32_t bits = 0;
        if (crossed(prev_ms, now_ms, 120.0f)) bits |= EVT_BTN_A_SHORT;   // cancel 2nd fall
        if (crossed(prev_ms, now_ms, 225.0f)) bits |= EVT_BTN_A_LONG;    // SOS
        // medication: confirm even-numbered reminders 20 s after they ring
        if (med_interval_s) {
            float a = prev_ms / 1000.0f, b = now_ms / 1000.0f;
            for (uint32_t k = 0; k < 1000; k += 2) {
                float t = med_first_s + k * (float)med_interval_s + 20.0f;
                if (t > b) break;
                if (t > a) bits |= EVT_BTN_B_SHORT;
            }
        }
        return bits;
    }

private:
    static bool crossed(uint32_t prev_ms, uint32_t now_ms, float at_ct) {
        if (now_ms <= prev_ms) return false;
        float a = prev_ms / 1000.0f, b = now_ms / 1000.0f;
        float base = floorf(b / kCycleS) * kCycleS;
        for (int c = 0; c < 2; ++c) {                      // check this and the previous cycle
            float t = base - c * kCycleS + at_ct;
            if (t > a && t <= b) return true;
        }
        return false;
    }
};

// ---- IMU generator (MPU6050 replacement), units g and deg/s ---------------------------
class MockImu {
public:
    explicit MockImu(uint32_t seed = 42) : rng_(seed) {}

    ImuSample sample(uint32_t t_ms) {
        const float t = t_ms / 1000.0f;
        const float ct = Scenario::cycleTime(t_ms);
        Vec3 a{0, 0, 0}, w{0, 0, 0};
        baseActivity(Scenario::mode(ct), t, a, w);

        // superimpose scripted transients
        if (ct >= 35.0f && ct < 36.2f) jump(ct - 35.0f, a, w);
        if (ct >= 55.0f && ct < 55.6f) fall(ct - 55.0f, kGWrist, kGLying1, a, w);
        if (ct >= 110.0f && ct < 110.6f) fall(ct - 110.0f, kGWrist, kGLying2, a, w);
        if (ct >= 132.0f && ct < 132.08f) slap(ct - 132.0f, a);

        a = a + Vec3{rng_.gauss(0.008f), rng_.gauss(0.008f), rng_.gauss(0.008f)};
        w = w + Vec3{rng_.gauss(0.5f), rng_.gauss(0.5f), rng_.gauss(0.5f)};
        return ImuSample{a.x, a.y, a.z, w.x, w.y, w.z, t_ms};
    }

private:
    // gravity direction in the sensor frame for the different postures
    static constexpr Vec3 kGWrist{0.15f, -0.25f, 0.956f};      // forearm on lap / hanging
    static constexpr Vec3 kGLying1{0.92f, 0.30f, -0.25f};      // after forward fall
    static constexpr Vec3 kGLying2{-0.35f, 0.90f, 0.26f};      // after lateral fall

    void baseActivity(ImuMode m, float t, Vec3& a, Vec3& w) {
        const float ct = fmodf(t, kCycleS);
        switch (m) {
            case ImuMode::REST:
                a = unit(kGWrist);
                break;
            case ImuMode::LYING:
                a = unit(ct < 110.0f ? kGLying1 : kGLying2);
                break;
            case ImuMode::WALK: {
                const float f = 0.9f, A = 0.45f;                     // arm swing 0.9 Hz, +-26 deg
                Vec3 axis = unit({1.0f, 0.2f, 0.0f});
                float th = A * sinf(2 * kPi * f * t);
                float dth = A * 2 * kPi * f * cosf(2 * kPi * f * t);
                a = rotate(unit(kGWrist), axis, th);
                float step = sinf(2 * kPi * 2 * f * t);
                step = (step > 0 ? 1.f : -1.f) * powf(fabsf(step), 1.5f) * 0.35f;  // heel strikes
                a = a + Vec3{0.2f, 0.9f, 0.3f} * step;
                w = axis * (dth * 57.2958f);
                break;
            }
            case ImuMode::GESTURE: {
                Vec3 axis = unit({0.3f, 1.0f, 0.2f});
                float th = 0.8f * sinf(2 * kPi * 1.5f * t) + 0.3f * sinf(2 * kPi * 0.4f * t);
                float dth = 0.8f * 2 * kPi * 1.5f * cosf(2 * kPi * 1.5f * t) + 0.3f * 2 * kPi * 0.4f * cosf(2 * kPi * 0.4f * t);
                a = rotate(unit(kGWrist), axis, th) + Vec3{rng_.gauss(0.12f), rng_.gauss(0.12f), rng_.gauss(0.12f)};
                w = axis * (dth * 57.2958f);
                break;
            }
        }
    }

    void jump(float r, Vec3& a, Vec3& w) {
        Vec3 g = unit(kGWrist);
        if (r < 0.25f)      a = g * 0.75f;                                // crouch
        else if (r < 0.45f) a = g * 1.8f;                                 // push-off
        else if (r < 0.75f) a = g * 0.12f;                                // flight (true free fall)
        else if (r < 0.81f) { float k = (r - 0.75f) / 0.06f; a = g * (1.0f + 2.8f * (1 - k)); }  // landing 3.8 g
        w = w + Vec3{rng_.gauss(40), rng_.gauss(40), rng_.gauss(40)};
    }

    void fall(float r, Vec3 g0, Vec3 g1, Vec3& a, Vec3& w) {
        g0 = unit(g0); g1 = unit(g1);
        Vec3 axis = unit(cross(g0, g1));
        float total = acosf(fmaxf(-1.f, fminf(1.f, dot(g0, g1))));
        const float ff = 0.35f, imp = 0.12f;
        if (r < ff) {                                                     // free fall + rotation
            float k = r / ff;
            a = rotate(g0, axis, k * total) * 0.2f;
            w = axis * (total / ff * 57.2958f);
        } else if (r < ff + imp) {                                        // impact: 40 ms peak, then ringing
            const float dt = r - ff;
            float spike = dt < 0.04f ? 4.5f - 10.0f * dt
                                     : 1.5f * expf(-30.0f * (dt - 0.04f)) * cosf(2 * kPi * 12.0f * (dt - 0.04f));
            a = g1 + unit(g1 + Vec3{0.4f, -0.3f, 0.5f}) * spike;
            w = Vec3{rng_.gauss(150), rng_.gauss(150), rng_.gauss(150)};
        } else {
            a = g1;
        }
    }

    void slap(float r, Vec3& a) {
        float k = r / 0.08f;
        a = a + unit({0.1f, 0.2f, 1.0f}) * (4.2f * expf(-4.0f * k));
    }

    Rng rng_;
};

// ---- PPG generator (MAX30102 replacement), raw counts at PPG_SAMPLE_HZ ------------------
class MockPpg {
public:
    explicit MockPpg(float fs = 25.0f, uint32_t seed = 7) : fs_(fs), rng_(seed) {}

    PpgSample sample(uint32_t t_ms, bool motion) {
        const float ct = Scenario::cycleTime(t_ms);
        Physiology p = Scenario::physiology(ct);
        phHr_ += p.hr / 60.0f / fs_;  phHr_ -= floorf(phHr_);
        phRr_ += p.rr / 60.0f / fs_;  phRr_ -= floorf(phRr_);
        float pulse = pulseShape(phHr_);
        float resp = sinf(2 * kPi * phRr_);
        float piIr = 0.012f * (1.0f + 0.15f * resp);              // perfusion index, resp. amplitude modulation
        float R = (110.0f - p.spo2) / 25.0f;                       // inverse of SpO2 = 110 - 25 R
        float piRed = R * piIr;
        float irDc = 120000.0f * (1.0f + 0.006f * resp);           // resp.-induced intensity variation
        float redDc = 95000.0f * (1.0f + 0.006f * resp);
        float ir = irDc * (1.0f - piIr * pulse) + rng_.gauss(40);
        float red = redDc * (1.0f - piRed * pulse) + rng_.gauss(40);
        if (motion) {                                              // large motion artefacts
            artefact_ += rng_.gauss(600);
            artefact_ *= 0.97f;
            float m = artefact_ + 2500.0f * sinf(2 * kPi * 1.8f * t_ms / 1000.0f);
            ir += m; red += 0.8f * m;
        }
        return PpgSample{(uint32_t)fmaxf(0.0f, red), (uint32_t)fmaxf(0.0f, ir), t_ms};
    }

    static float pulseShape(float ph) {                            // systolic peak + dicrotic wave
        float a = (ph - 0.15f) / 0.08f, b = (ph - 0.45f) / 0.10f;
        return expf(-a * a) + 0.3f * expf(-b * b);
    }

private:
    float fs_, phHr_ = 0, phRr_ = 0, artefact_ = 0;
    Rng rng_;
};

// ---- battery: slow discharge from 4.15 V ---------------------------------------------
inline float mockBatteryVoltage(uint32_t t_ms) {
    float v = 4.15f - 0.00002f * (t_ms / 1000.0f);
    return v < 3.35f ? 3.35f : v;
}

}  // namespace sim
