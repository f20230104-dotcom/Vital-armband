// =============================================================================
// types.h - Shared data types, message payloads and RTOS handles
// =============================================================================
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <math.h>

// ---- sensor samples -----------------------------------------------------------
struct ImuSample {
    float ax, ay, az;      // g
    float gx, gy, gz;      // deg/s
    uint32_t t_ms;
};

struct PpgSample {
    uint32_t red;          // raw 18-bit ADC counts
    uint32_t ir;
    uint32_t t_ms;
};

// ---- events -------------------------------------------------------------------
enum class EventType : uint8_t {
    TELEMETRY = 0,
    FALL,
    FALL_CANCELLED,
    SOS,
    MED_REMINDER,
    MED_TAKEN,
    MED_MISSED,
    VITALS_ABNORMAL,
    LOW_BATTERY,
    BOOT,
};

enum Priority : uint8_t {
    PRIO_LOW = 0,          // informational, batched with the next heartbeat
    PRIO_NORMAL = 1,
    PRIO_HIGH = 2,         // missed medication, abnormal vitals
    PRIO_CRITICAL = 3,     // fall, SOS
};

inline const char* eventName(EventType e) {
    switch (e) {
        case EventType::TELEMETRY:       return "TELEMETRY";
        case EventType::FALL:            return "FALL";
        case EventType::FALL_CANCELLED:  return "FALL_CANCELLED";
        case EventType::SOS:             return "SOS";
        case EventType::MED_REMINDER:    return "MED_REMINDER";
        case EventType::MED_TAKEN:       return "MED_TAKEN";
        case EventType::MED_MISSED:      return "MED_MISSED";
        case EventType::VITALS_ABNORMAL: return "VITALS_ABNORMAL";
        case EventType::LOW_BATTERY:     return "LOW_BATTERY";
        case EventType::BOOT:            return "BOOT";
    }
    return "UNKNOWN";
}

// Message posted to the alert queues (copied by value, keep it small).
struct AlertMsg {
    EventType type;
    uint8_t   priority;
    uint32_t  t_ms;          // millis() when raised
    uint32_t  unix_time;     // RTC time when raised (0 = unknown)
    float     fall_prob;     // NAN if not applicable
    float     peak_g;        // NAN if not applicable
    char      detail[64];
};

// Latest vitals, published by vitals_task through a 1-slot mailbox queue.
struct VitalsSnapshot {
    float   hr;              // bpm, NAN = unknown
    float   spo2;            // %
    float   rr;              // breaths/min
    bool    contact;         // skin contact detected
    bool    motion_gated;    // readings frozen because the arm is moving
    uint32_t t_ms;
};

// ---- fall-detector state machine ---------------------------------------------
enum class FallState : uint8_t { IDLE, FREE_FALL, IMPACT, POST_IMPACT, INFERENCE, COUNTDOWN, COOLDOWN };

inline const char* fallStateName(FallState s) {
    switch (s) {
        case FallState::IDLE:        return "IDLE";
        case FallState::FREE_FALL:   return "FREE_FALL";
        case FallState::IMPACT:      return "IMPACT";
        case FallState::POST_IMPACT: return "POST_IMPACT";
        case FallState::INFERENCE:   return "INFERENCE";
        case FallState::COUNTDOWN:   return "COUNTDOWN";
        case FallState::COOLDOWN:    return "COOLDOWN";
    }
    return "?";
}

// ---- local annunciator patterns -----------------------------------------------
enum class AlarmPattern : uint8_t { NONE, FALL_COUNTDOWN, MED_REMINDER, SOS_ACK, ABNORMAL, CONFIRM };

// ---- event-group bits ----------------------------------------------------------
#define EVT_MOTION_ACTIVE    (1u << 0)   // set by motion_task, read by vitals_task (PPG gate)
#define EVT_BTN_A_SHORT      (1u << 1)
#define EVT_BTN_A_LONG       (1u << 2)
#define EVT_BTN_B_SHORT      (1u << 3)
#define EVT_FALL_ALARM       (1u << 4)   // fall countdown running
#define EVT_MED_PENDING      (1u << 5)   // waiting for medication confirmation
#define EVT_RTC_ALARM        (1u << 6)   // DS3231 INT fired
#define EVT_BTN_ANY_SHORT    (EVT_BTN_A_SHORT | EVT_BTN_B_SHORT)

// ---- RTOS handles (defined in main.cpp, only visible to firmware builds) -------
#if defined(ARDUINO)
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/event_groups.h>
#include <freertos/semphr.h>

extern QueueHandle_t      g_alertQueueHigh;   // PRIO_HIGH / PRIO_CRITICAL
extern QueueHandle_t      g_alertQueueLow;    // PRIO_LOW / PRIO_NORMAL
extern QueueSetHandle_t   g_alertQueueSet;
extern QueueHandle_t      g_vitalsMailbox;    // length 1, xQueueOverwrite / xQueuePeek
extern EventGroupHandle_t g_events;
extern SemaphoreHandle_t  g_i2cMutex;

// Enqueue an alert from any task (never blocks the caller for long).
bool raiseAlert(EventType type, uint8_t priority, const char* detail = nullptr,
                float fall_prob = NAN, float peak_g = NAN);

// Task entry points
void motionTask(void*);
void vitalsTask(void*);
void reminderTask(void*);
void alertTask(void*);
FallState motionCurrentState();
#endif
