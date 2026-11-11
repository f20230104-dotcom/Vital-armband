// =============================================================================
// vitals_task.cpp - MAX30102 PPG -> HR / SpO2 / RR with motion gate and persistence
//
//  * Motion gate : samples are discarded while EVT_MOTION_ACTIVE is set (motion_task)
//  * RR          : 0.1-0.5 Hz band-pass on the PPG baseline (dsp_filters.cpp)
//  * Persistence : an alert is raised only if a vital stays abnormal for
//                  VITALS_ABNORMAL_PERSIST_S consecutive valid seconds
// =============================================================================
#include <Arduino.h>
#include <math.h>

#include "config.h"
#include "types.h"
#include "hardware_hal.h"
#include "ppg_processor.h"

namespace {

struct Persist {
    const char* name;
    int abnormal = 0;       // consecutive abnormal seconds
    int invalid = 0;        // consecutive seconds without a valid reading

    void update(bool valid, bool isAbnormal) {
        if (!valid) {                           // gated / settling: pause, but not forever
            if (++invalid > 10) abnormal = 0;
            return;
        }
        invalid = 0;
        abnormal = isAbnormal ? abnormal + 1 : 0;
    }
};

}  // namespace

void vitalsTask(void*) {
    static PpgProcessor proc((float)PPG_SAMPLE_HZ, RR_DECIMATION, PPG_CONTACT_MIN_IR);
    PpgSample buf[32];
    VitalsSnapshot snap{NAN, NAN, NAN, false, false, 0};
    Persist hr{"HR"}, spo2{"SpO2"}, rr{"RR"};
    bool latched = false;
    uint32_t lastAlertMs = 0, lastSecond = millis(), lastLog = 0;

    TickType_t last = xTaskGetTickCount();
    wdtSubscribe();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(40));          // FIFO holds 32 samples (>1 s @25 Hz)
        wdtFeed();
        const bool motion = xEventGroupGetBits(g_events) & EVT_MOTION_ACTIVE;
        const int n = hal_readPpg(buf, 32);
        for (int i = 0; i < n; ++i) proc.push(buf[i].red, buf[i].ir, motion);

        if (millis() - lastSecond < 1000) continue;
        lastSecond += 1000;

        const VitalsEstimate e = proc.estimate();
        if (!e.contact && !e.gated) {
            snap.hr = snap.spo2 = snap.rr = NAN;
        } else {                                             // keep last good values while gated
            if (e.hrValid) snap.hr = e.hr;
            if (e.spo2Valid) snap.spo2 = e.spo2;
            if (e.rrValid) snap.rr = e.rr;
        }
        snap.contact = e.contact;
        snap.motion_gated = e.gated;
        snap.t_ms = millis();
        xQueueOverwrite(g_vitalsMailbox, &snap);

        hr.update(e.hrValid, e.hr < HR_LOW_BPM || e.hr > HR_HIGH_BPM);
        spo2.update(e.spo2Valid, e.spo2 < SPO2_LOW_PCT);
        rr.update(e.rrValid, e.rr < RR_LOW_BPM || e.rr > RR_HIGH_BPM);

        const Persist* worst = nullptr;
        for (const Persist* p : {&spo2, &hr, &rr})
            if (p->abnormal >= VITALS_ABNORMAL_PERSIST_S) { worst = p; break; }

        if (worst && (!latched || millis() - lastAlertMs > VITALS_ALERT_REARM_S * 1000UL)) {
            char detail[64];
            snprintf(detail, sizeof(detail), "%s abnormal for %d s (HR %.0f, SpO2 %.1f, RR %.0f)", worst->name,
                     worst->abnormal, snap.hr, snap.spo2, snap.rr);
            raiseAlert(EventType::VITALS_ABNORMAL, PRIO_HIGH, detail);
            hal_alarmStart(AlarmPattern::ABNORMAL);
            latched = true;
            lastAlertMs = millis();
        } else if (latched && hr.abnormal == 0 && spo2.abnormal == 0 && rr.abnormal == 0) {
            latched = false;
            hal_alarmStop(AlarmPattern::ABNORMAL);
            Serial.println("[vitals] vitals back to normal");
        }

        if (millis() - lastLog >= 5000) {
            lastLog = millis();
            Serial.printf("[vitals] HR %5.1f%s SpO2 %5.1f%s RR %4.1f%s contact=%d gated=%d abn(s) hr=%d spo2=%d rr=%d\n",
                          snap.hr, e.hrValid ? "" : "*", snap.spo2, e.spo2Valid ? "" : "*", snap.rr,
                          e.rrValid ? "" : "*", e.contact, e.gated, hr.abnormal, spo2.abnormal, rr.abnormal);
        }
    }
}
