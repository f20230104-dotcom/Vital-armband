// =============================================================================
// reminder_task.cpp - medication alarms and confirmation countdown
//
// The DS3231 alarm 1 is armed for the next dose in MED_SCHEDULE; its INT pin (GPIO33)
// fires at that time. Without a DS3231 (USE_DS3231 false) the HAL runs the same daily alarm
// in software on the NTP-synced ESP32 clock. The wristband beeps/vibrates and waits for button B. If the dose
// is not confirmed within MED_CONFIRM_TIMEOUT_S (5 min) a MED_MISSED alert is sent.
// In SIMULATION_MODE the alarm comes from an accelerated schedule (hardware_hal.cpp).
// =============================================================================
#include <Arduino.h>

#include "config.h"
#include "types.h"
#include "hardware_hal.h"

namespace {

struct Med { uint8_t hour, minute; const char* name; };
const Med kMeds[] = MED_SCHEDULE;
constexpr int kMedCount = sizeof(kMeds) / sizeof(kMeds[0]);
constexpr uint32_t kNagOnMs = 20000;          // ring for 20 s, then re-nag every MED_RENAG_PERIOD_S

int nextDoseIndex(uint32_t unix) {
    uint8_t h, m;
    hal_rtcHourMinute(unix, h, m);
    const int now = h * 60 + m;
    for (int i = 0; i < kMedCount; ++i)
        if (kMeds[i].hour * 60 + kMeds[i].minute > now) return i;
    return 0;                                 // tomorrow's first dose
}

}  // namespace

void reminderTask(void*) {
    int doseIdx = nextDoseIndex(hal_rtcNow());
    int simCounter = 0;
    if (!hal_isSimulated(Sensor::RTC)) {
        hal_rtcSetDailyAlarm(kMeds[doseIdx].hour, kMeds[doseIdx].minute);
        Serial.printf("[reminder] next dose %02u:%02u %s\n", kMeds[doseIdx].hour, kMeds[doseIdx].minute,
                      kMeds[doseIdx].name);
    }

    bool pending = false, ringing = false;
    const Med* current = nullptr;
    uint32_t pendingSince = 0, lastNag = 0;
    [[maybe_unused]] uint32_t lastRearmCheck = 0;
    char detail[64];

    wdtSubscribe();
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(250));
        wdtFeed();

        if (hal_rtcAlarmFired()) {
            if (pending) {                    // previous dose never confirmed -> escalate now
                snprintf(detail, sizeof(detail), "%02u:%02u %s not confirmed", current->hour, current->minute,
                         current->name);
                raiseAlert(EventType::MED_MISSED, PRIO_HIGH, detail);
            }
            if (hal_isSimulated(Sensor::RTC)) {
                current = &kMeds[simCounter++ % kMedCount];
            } else {
                current = &kMeds[doseIdx];
                doseIdx = (doseIdx + 1) % kMedCount;
                hal_rtcSetDailyAlarm(kMeds[doseIdx].hour, kMeds[doseIdx].minute);
            }
            pending = ringing = true;
            pendingSince = lastNag = millis();
            xEventGroupClearBits(g_events, EVT_BTN_B_SHORT);
            xEventGroupSetBits(g_events, EVT_MED_PENDING);
            hal_alarmStart(AlarmPattern::MED_REMINDER);
            snprintf(detail, sizeof(detail), "%02u:%02u %s - press B to confirm", current->hour, current->minute,
                     current->name);
            Serial.printf("[reminder] %s\n", detail);
            raiseAlert(EventType::MED_REMINDER, PRIO_LOW, detail);
        }

#if !USE_DS3231
        // The software clock can jump when NTP first syncs (it starts from the build time), so
        // re-pick the next dose once a minute while nothing is pending.
        if (!pending && !hal_isSimulated(Sensor::RTC) && millis() - lastRearmCheck > 60000) {
            lastRearmCheck = millis();
            const int idx = nextDoseIndex(hal_rtcNow());
            if (idx != doseIdx) {
                doseIdx = idx;
                hal_rtcSetDailyAlarm(kMeds[doseIdx].hour, kMeds[doseIdx].minute);
                Serial.printf("[reminder] clock changed, next dose %02u:%02u %s\n", kMeds[doseIdx].hour,
                              kMeds[doseIdx].minute, kMeds[doseIdx].name);
            }
        }
#endif

        if (!pending) continue;
        const uint32_t elapsed = millis() - pendingSince;

        if (xEventGroupClearBits(g_events, EVT_BTN_B_SHORT) & EVT_BTN_B_SHORT) {
            snprintf(detail, sizeof(detail), "%s confirmed after %lu s", current->name, (unsigned long)(elapsed / 1000));
            Serial.printf("[reminder] %s\n", detail);
            hal_alarmStop(AlarmPattern::MED_REMINDER);
            hal_alarmStart(AlarmPattern::CONFIRM);
            raiseAlert(EventType::MED_TAKEN, PRIO_LOW, detail);
            pending = ringing = false;
            xEventGroupClearBits(g_events, EVT_MED_PENDING);
        } else if (elapsed >= MED_CONFIRM_TIMEOUT_S * 1000UL) {
            snprintf(detail, sizeof(detail), "%02u:%02u %s not confirmed within %d s", current->hour,
                     current->minute, current->name, MED_CONFIRM_TIMEOUT_S);
            Serial.printf("[reminder] MISSED: %s\n", detail);
            hal_alarmStop(AlarmPattern::MED_REMINDER);
            raiseAlert(EventType::MED_MISSED, PRIO_HIGH, detail);
            pending = ringing = false;
            xEventGroupClearBits(g_events, EVT_MED_PENDING);
        } else if (ringing && millis() - lastNag >= kNagOnMs) {
            hal_alarmStop(AlarmPattern::MED_REMINDER);       // pause the nag to save battery
            ringing = false;
        } else if (!ringing && millis() - lastNag >= MED_RENAG_PERIOD_S * 1000UL) {
            hal_alarmStart(AlarmPattern::MED_REMINDER);
            ringing = true;
            lastNag = millis();
        }
    }
}
