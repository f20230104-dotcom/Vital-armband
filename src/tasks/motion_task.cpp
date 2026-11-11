// =============================================================================
// motion_task.cpp - 50 Hz IMU polling, fall state machine, TFLite inference trigger
//
// Also owns the PPG motion gate: EVT_MOTION_ACTIVE is set while the arm moves so
// vitals_task can discard corrupted PPG samples.
// =============================================================================
#include <Arduino.h>

#include "config.h"
#include "types.h"
#include "hardware_hal.h"
#include "fall_detector.h"
#include "fall_classifier.h"

static FallDetector s_detector;
static volatile FallState s_state = FallState::IDLE;

FallState motionCurrentState() { return s_state; }

void motionTask(void*) {
    if (fallClassifierReady()) {
        s_detector.setClassifier(&fallClassifierPredict, nullptr);
        Serial.println("[motion] classifier: TFLite Micro int8 CNN");
    } else {
        s_detector.setClassifier(&FallDetector::heuristicScore, nullptr);
        Serial.println("[motion] classifier: rule-based fallback");
    }

    const TickType_t period = pdMS_TO_TICKS(1000 / MOTION_SAMPLE_HZ);
    TickType_t last = xTaskGetTickCount();
    ImuSample s;
    uint32_t readErrors = 0;
    bool gate = false;

    wdtSubscribe();
    for (;;) {
        vTaskDelayUntil(&last, period);
        wdtFeed();
        if (!hal_readImu(s)) {
            if (readErrors++ % MOTION_SAMPLE_HZ == 0) Serial.println("[motion] IMU read failed");
            continue;
        }

        bool cancel = false;
        if (s_detector.state() == FallState::COUNTDOWN) {
            // xEventGroupClearBits returns the bits *before* clearing
            cancel = xEventGroupClearBits(g_events, EVT_BTN_ANY_SHORT) & EVT_BTN_ANY_SHORT;
        }

        const FallDetector::Output o = s_detector.update(s, cancel);

        if (o.motionActive != gate) {
            gate = o.motionActive;
            if (gate) xEventGroupSetBits(g_events, EVT_MOTION_ACTIVE);
            else xEventGroupClearBits(g_events, EVT_MOTION_ACTIVE);
        }

        if (o.stateChanged) {
            Serial.printf("[motion] t=%lus -> %s: %s\n", (unsigned long)(s.t_ms / 1000),
                          fallStateName(s_detector.state()), o.note ? o.note : "");
        }
        if (o.prob >= 0) {
            Serial.printf("[motion] inference p(fall)=%.3f in %lu us (peak %.2f g, free fall %.0f ms)\n",
                          o.prob, (unsigned long)fallClassifierLastMicros(), s_detector.lastPeakG(),
                          s_detector.lastFreeFallMs());
        }
        if (o.startCountdown) {
            xEventGroupClearBits(g_events, EVT_BTN_ANY_SHORT);       // ignore stale presses
            xEventGroupSetBits(g_events, EVT_FALL_ALARM);
            hal_alarmStart(AlarmPattern::FALL_COUNTDOWN);
            Serial.printf("[motion] FALL suspected - press any button within %d s to cancel\n", FALL_COUNTDOWN_S);
        }
        if (o.fallConfirmed || o.cancelled) {
            hal_alarmStop(AlarmPattern::FALL_COUNTDOWN);
            xEventGroupClearBits(g_events, EVT_FALL_ALARM);
            char detail[64];
            snprintf(detail, sizeof(detail), "free fall %.0f ms, impact %.1f g%s", s_detector.lastFreeFallMs(),
                     s_detector.lastPeakG(), o.cancelled ? ", cancelled by wearer" : ", no response");
            if (o.fallConfirmed) {
                raiseAlert(EventType::FALL, PRIO_CRITICAL, detail, s_detector.lastProbability(), s_detector.lastPeakG());
            } else {
                hal_alarmStart(AlarmPattern::CONFIRM);
                raiseAlert(EventType::FALL_CANCELLED, PRIO_LOW, detail, s_detector.lastProbability(),
                           s_detector.lastPeakG());
            }
        }
        s_state = s_detector.state();
    }
}
