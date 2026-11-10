// =============================================================================
// host_sim.cpp - Run the firmware's sensing logic on your PC, faster than real time.
//
// Uses the exact same sources as the ESP32 build (mock_sensors.h, fall_detector.cpp,
// ppg_processor.cpp, dsp_filters.cpp) and replays the SIMULATION_MODE scenario, printing
// the event timeline the wristband would send. Optionally dumps every model window to a
// CSV so ml_training/verify_sim_windows.py can score them with the real int8 TFLite model.
//
//   c++ -std=c++17 -O2 -Iinclude tools/host_sim.cpp src/drivers/dsp_filters.cpp \
//       src/drivers/ppg_processor.cpp src/drivers/fall_detector.cpp -o host_sim
//   ./host_sim [seconds=480] [windows.csv]
// =============================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "config.h"
#include "fall_detector.h"
#include "mock_sensors.h"
#include "ppg_processor.h"

namespace {
std::vector<std::vector<float>> g_windows;
int g_events[16] = {0};

float recordingClassifier(const float* w, void*) {
    g_windows.emplace_back(w, w + FALL_WINDOW_SAMPLES * 6);
    return FallDetector::heuristicScore(w);
}

void event(uint32_t t_ms, EventType e, const char* detail) {
    ++g_events[(int)e];
    std::printf("%7.2fs  >>> %-15s %s\n", t_ms / 1000.0, eventName(e), detail);
}
}  // namespace

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 480;
    const char* dump = argc > 2 ? argv[2] : nullptr;

    sim::MockImu imu;
    sim::MockPpg ppg((float)PPG_SAMPLE_HZ);
    FallDetector det;
    det.setClassifier(&recordingClassifier, nullptr);
    PpgProcessor proc((float)PPG_SAMPLE_HZ, RR_DECIMATION, PPG_CONTACT_MIN_IR);

    uint32_t pendingBtn = 0, prev = 0;
    int abnSpo2 = 0, abnHr = 0, abnRr = 0;
    bool vitalsLatched = false;
    bool medPending = false;
    uint32_t medSince = 0;
    char buf[128];

    for (uint32_t t = 0; t <= (uint32_t)seconds * 1000; t += 1000 / MOTION_SAMPLE_HZ) {
        pendingBtn |= sim::Scenario::buttons(prev, t, SIM_MED_FIRST_S, SIM_MED_INTERVAL_S);
        prev = t;

        // ---- motion / fall (motion_task) ------------------------------------------
        bool cancel = false;
        if (det.state() == FallState::COUNTDOWN && (pendingBtn & EVT_BTN_ANY_SHORT)) {
            cancel = true;
            pendingBtn &= ~EVT_BTN_ANY_SHORT;
        }
        const FallDetector::Output o = det.update(imu.sample(t), cancel);
        if (o.stateChanged) std::printf("%7.2fs  [fsm] %-11s %s\n", t / 1000.0, fallStateName(det.state()), o.note);
        if (o.prob >= 0) std::printf("%7.2fs  [fsm] p(fall)=%.2f peak=%.2f g free-fall=%.0f ms\n", t / 1000.0, o.prob,
                                     det.lastPeakG(), det.lastFreeFallMs());
        if (o.startCountdown) pendingBtn &= ~EVT_BTN_ANY_SHORT;
        if (o.fallConfirmed) event(t, EventType::FALL, "countdown expired");
        if (o.cancelled) event(t, EventType::FALL_CANCELLED, "wearer pressed a button");

        // ---- SOS (supervisor) -------------------------------------------------------
        if (pendingBtn & EVT_BTN_A_LONG) { pendingBtn &= ~EVT_BTN_A_LONG; event(t, EventType::SOS, "long press"); }

        // ---- PPG @25 Hz (vitals_task) ---------------------------------------------
        if (t % (1000 / PPG_SAMPLE_HZ) == 0) {
            const sim::ImuMode m = sim::Scenario::mode(sim::Scenario::cycleTime(t));
            const bool moving = m == sim::ImuMode::WALK || m == sim::ImuMode::GESTURE;
            const PpgSample s = ppg.sample(t, moving);
            proc.push(s.red, s.ir, o.motionActive);
        }
        if (t % 1000 == 0 && t > 0) {
            const VitalsEstimate e = proc.estimate();
            auto upd = [](int& c, bool valid, bool abn) { if (valid) c = abn ? c + 1 : 0; };
            upd(abnSpo2, e.spo2Valid, e.spo2 < SPO2_LOW_PCT);
            upd(abnHr, e.hrValid, e.hr < HR_LOW_BPM || e.hr > HR_HIGH_BPM);
            upd(abnRr, e.rrValid, e.rr < RR_LOW_BPM || e.rr > RR_HIGH_BPM);
            const bool abn = abnSpo2 >= VITALS_ABNORMAL_PERSIST_S || abnHr >= VITALS_ABNORMAL_PERSIST_S ||
                             abnRr >= VITALS_ABNORMAL_PERSIST_S;
            if (abn && !vitalsLatched) {
                std::snprintf(buf, sizeof buf, "HR %.0f SpO2 %.1f RR %.0f", e.hr, e.spo2, e.rr);
                event(t, EventType::VITALS_ABNORMAL, buf);
                vitalsLatched = true;
            } else if (!abn && vitalsLatched && !abnSpo2 && !abnHr && !abnRr) {
                vitalsLatched = false;
            }
            if (t % 5000 == 0) {
                const float ct = sim::Scenario::cycleTime(t);
                const sim::Physiology p = sim::Scenario::physiology(ct);
                std::printf("%7.2fs  [vitals] HR %5.1f%s (true %5.1f)  SpO2 %5.1f%s (true %5.1f)  RR %4.1f%s (true %4.1f)%s\n",
                            t / 1000.0, e.hr, e.hrValid ? " " : "*", p.hr, e.spo2, e.spo2Valid ? " " : "*", p.spo2,
                            e.rr, e.rrValid ? " " : "*", p.rr, e.gated ? "  [motion-gated]" : "");
            }
        }

        // ---- medication (reminder_task, accelerated schedule) ----------------------
        if (t % 1000 == 0) {
            const uint32_t ts = t / 1000;
            if (ts >= SIM_MED_FIRST_S && (ts - SIM_MED_FIRST_S) % SIM_MED_INTERVAL_S == 0) {
                if (medPending) event(t, EventType::MED_MISSED, "previous dose");
                event(t, EventType::MED_REMINDER, "dose due");
                medPending = true;
                medSince = t;
                pendingBtn &= ~EVT_BTN_B_SHORT;
            }
        }
        if (medPending && (pendingBtn & EVT_BTN_B_SHORT)) {
            pendingBtn &= ~EVT_BTN_B_SHORT;
            event(t, EventType::MED_TAKEN, "confirmed with button B");
            medPending = false;
        } else if (medPending && t - medSince >= MED_CONFIRM_TIMEOUT_S * 1000u) {
            event(t, EventType::MED_MISSED, "no confirmation");
            medPending = false;
        }
    }

    std::printf("\nSummary over %d s:", seconds);
    for (int e = 1; e <= (int)EventType::BOOT; ++e)
        if (g_events[e]) std::printf("  %s=%d", eventName((EventType)e), g_events[e]);
    std::printf("\nClassifier invocations: %zu\n", g_windows.size());

    if (dump) {
        FILE* f = std::fopen(dump, "w");
        if (!f) { std::perror(dump); return 1; }
        for (const auto& w : g_windows) {
            for (size_t i = 0; i < w.size(); ++i) std::fprintf(f, i ? ",%.5f" : "%.5f", w[i]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        std::printf("Wrote %zu windows to %s\n", g_windows.size(), dump);
    }
    return 0;
}
