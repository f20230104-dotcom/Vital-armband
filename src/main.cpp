// =============================================================================
// main.cpp - Vital wristband firmware entry point
//
//  Core 1: motion_task (prio 5, 50 Hz IMU + fall FSM + TFLM)  vitals_task (prio 4, PPG)
//  Core 0: reminder_task (prio 3, DS3231 alarms)               alert_task (prio 2, WiFi)
//  loop() : supervisor - buttons/SOS, battery monitor, health stats
// =============================================================================
#include <Arduino.h>
#include <esp_system.h>

#include "config.h"
#include "types.h"
#include "hardware_hal.h"
#include "fall_classifier.h"

// ---- shared RTOS objects (declared in types.h) ------------------------------------
QueueHandle_t      g_alertQueueHigh = nullptr;
QueueHandle_t      g_alertQueueLow = nullptr;
QueueSetHandle_t   g_alertQueueSet = nullptr;
QueueHandle_t      g_vitalsMailbox = nullptr;
EventGroupHandle_t g_events = nullptr;
SemaphoreHandle_t  g_i2cMutex = nullptr;

static TaskHandle_t s_tasks[4];
static const char* const kTaskNames[4] = {"motion", "vitals", "reminder", "alert"};

bool raiseAlert(EventType type, uint8_t priority, const char* detail, float fall_prob, float peak_g) {
    AlertMsg m{};
    m.type = type;
    m.priority = priority;
    m.t_ms = millis();
    m.unix_time = hal_rtcNow();
    m.fall_prob = fall_prob;
    m.peak_g = peak_g;
    if (detail) strlcpy(m.detail, detail, sizeof(m.detail));
    QueueHandle_t q = priority >= PRIO_HIGH ? g_alertQueueHigh : g_alertQueueLow;
    const bool ok = xQueueSend(q, &m, pdMS_TO_TICKS(20)) == pdTRUE;
    Serial.printf("[event] %-15s prio=%u %s%s\n", eventName(type), priority, detail ? detail : "",
                  ok ? "" : "  !! queue full");
    if (ok && priority >= PRIO_HIGH) hal_eepromLog(type, m.unix_time);
    return ok;
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.printf("\n=== Vital wristband fw %s  device %s  %s ===\n", FW_VERSION, DEVICE_ID,
                  SIMULATION_MODE ? "SIMULATION_MODE" : "HARDWARE MODE");
    Serial.printf("reset reason %d, heap %u B\n", (int)esp_reset_reason(), ESP.getFreeHeap());

    g_i2cMutex = xSemaphoreCreateMutex();
    g_events = xEventGroupCreate();
    g_vitalsMailbox = xQueueCreate(1, sizeof(VitalsSnapshot));
    g_alertQueueHigh = xQueueCreate(ALERT_QUEUE_LEN_HIGH, sizeof(AlertMsg));
    g_alertQueueLow = xQueueCreate(ALERT_QUEUE_LEN_LOW, sizeof(AlertMsg));
    g_alertQueueSet = xQueueCreateSet(ALERT_QUEUE_LEN_HIGH + ALERT_QUEUE_LEN_LOW);
    configASSERT(g_i2cMutex && g_events && g_vitalsMailbox && g_alertQueueHigh && g_alertQueueLow && g_alertQueueSet);
    xQueueAddToSet(g_alertQueueHigh, g_alertQueueSet);
    xQueueAddToSet(g_alertQueueLow, g_alertQueueSet);

    if (!hal_init()) {
        Serial.println("[main] HAL init failed - halting");
        for (;;) delay(1000);
    }

    if (FALL_USE_TFLM && fallClassifierInit()) {
        float pf = 0, pa = 0;
        const bool ok = fallClassifierSelfTest(&pf, &pa);
        Serial.printf("[main] TFLM self-test %s: p(fall|fall)=%.3f p(fall|adl)=%.3f\n", ok ? "PASSED" : "FAILED", pf, pa);
    } else {
        Serial.println("[main] TFLM unavailable - using rule-based fall score");
    }

    xTaskCreatePinnedToCore(motionTask, kTaskNames[0], MOTION_TASK_STACK, nullptr, MOTION_TASK_PRIO, &s_tasks[0],
                            MOTION_TASK_CORE);
    xTaskCreatePinnedToCore(vitalsTask, kTaskNames[1], VITALS_TASK_STACK, nullptr, VITALS_TASK_PRIO, &s_tasks[1],
                            VITALS_TASK_CORE);
    xTaskCreatePinnedToCore(reminderTask, kTaskNames[2], REMINDER_TASK_STACK, nullptr, REMINDER_TASK_PRIO,
                            &s_tasks[2], REMINDER_TASK_CORE);
    xTaskCreatePinnedToCore(alertTask, kTaskNames[3], ALERT_TASK_STACK, nullptr, ALERT_TASK_PRIO, &s_tasks[3],
                            ALERT_TASK_CORE);

    char detail[64];
    snprintf(detail, sizeof(detail), "fw %s, %s", FW_VERSION, hal_sensorSummary());
    raiseAlert(EventType::BOOT, PRIO_NORMAL, detail);
}

// Supervisor: runs in Arduino's loopTask (core 1, prio 1)
void loop() {
    static uint32_t lastBatt = 0, lastStats = 0;
    static bool lowBattSent = false;
    const uint32_t now = millis();

    hal_pollButtons();

    // SOS: long press on button A
    if (xEventGroupClearBits(g_events, EVT_BTN_A_LONG) & EVT_BTN_A_LONG) {
        hal_alarmStart(AlarmPattern::SOS_ACK);
        raiseAlert(EventType::SOS, PRIO_CRITICAL, "SOS button held 2 s");
    }

    if (now - lastBatt >= BATT_CHECK_PERIOD_S * 1000UL || lastBatt == 0) {
        lastBatt = now;
        const float v = hal_batteryVoltage();
        const uint8_t pct = hal_batteryPercent(v);
        if (!lowBattSent && pct < BATT_LOW_PCT) {
            char d[48];
            snprintf(d, sizeof(d), "battery %.2f V (%u %%)", v, pct);
            raiseAlert(EventType::LOW_BATTERY, PRIO_NORMAL, d);
            lowBattSent = true;
        } else if (lowBattSent && pct > BATT_LOW_PCT + 5) {
            lowBattSent = false;                          // hysteresis: charged again
        }
    }

    if (now - lastStats >= 30000) {
        lastStats = now;
        Serial.printf("[sys] up %lus heap %u (min %u) fall=%s stack-free:", (unsigned long)(now / 1000),
                      ESP.getFreeHeap(), ESP.getMinFreeHeap(), fallStateName(motionCurrentState()));
        for (int i = 0; i < 4; ++i)
            Serial.printf(" %s=%u", kTaskNames[i], (unsigned)uxTaskGetStackHighWaterMark(s_tasks[i]));
        Serial.println();
    }
    vTaskDelay(pdMS_TO_TICKS(20));
}
