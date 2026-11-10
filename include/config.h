// =============================================================================
// config.h - Vital wristband: build-time configuration
// Pin map, I2C addresses, detection thresholds, task layout, WiFi / webhook.
// Put real credentials in include/secrets.h (git-ignored), see secrets.example.h.
// =============================================================================
#pragma once

#include <stdint.h>

// -----------------------------------------------------------------------------
// Build mode
// -----------------------------------------------------------------------------
// true  : all sensors are replaced by realistic generators (mock_sensors.h) driven by a
//         scripted scenario (walk, jump, falls, hypoxia, SOS, medication). The full
//         firmware still runs: state machine, TFLite inference, DSP, RTOS queues, WiFi POST.
// false : real MPU6050 / MAX30102 / DS3231 drivers (with per-sensor fallback to mock if a
//         chip does not answer on the bus and AUTO_FALLBACK_TO_MOCK is true).
#ifndef SIMULATION_MODE
#define SIMULATION_MODE true
#endif
#define AUTO_FALLBACK_TO_MOCK true

#define FW_VERSION          "1.0.0"
#define DEVICE_ID           "WB-01"

// -----------------------------------------------------------------------------
// WiFi / webhook  (override in secrets.h)
// -----------------------------------------------------------------------------
#if defined(__has_include)
#if __has_include("secrets.h")
#include "secrets.h"
#endif
#endif
#ifndef WIFI_SSID
#define WIFI_SSID           "YOUR_WIFI_SSID"
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD       "YOUR_WIFI_PASSWORD"
#endif
#ifndef WEBHOOK_URL
// IP of the PC running `python dashboard/server.py` (or app.py) on the same LAN.
#define WEBHOOK_URL         "http://192.168.1.100:8000/api/telemetry"
#endif
#ifndef WEBHOOK_API_KEY
#define WEBHOOK_API_KEY     ""          // matches WRISTBAND_API_KEY on the server ("" = off)
#endif

#define WIFI_CONNECT_TIMEOUT_MS   10000
#define HTTP_TIMEOUT_MS           5000
#define HTTP_MAX_RETRIES          3
#define RETRY_BUFFER_LEN          16    // undelivered messages kept for the next WiFi window

// Heartbeat telemetry (vitals snapshot). WiFi is off between windows to save battery.
#if SIMULATION_MODE
#define TELEMETRY_PERIOD_S        15
#else
#define TELEMETRY_PERIOD_S        300
#endif
// Messages of priority >= this are sent immediately; lower ones wait for the next heartbeat.
#define IMMEDIATE_PRIORITY        PRIO_NORMAL

// -----------------------------------------------------------------------------
// Pins (ESP32-WROOM-32)
// -----------------------------------------------------------------------------
#define PIN_I2C_SDA         21
#define PIN_I2C_SCL         22
#define I2C_CLOCK_HZ        400000

#define PIN_MPU_INT         27
#define PIN_RTC_INT         33      // DS3231 INT/SQW, open-drain, active LOW
#define PIN_MAX_INT         32      // MAX30102 INT, open-drain, active LOW

#define PIN_BTN_A           4       // short: cancel fall countdown   long (2 s): SOS
#define PIN_BTN_B           13      // short: confirm medication / cancel fall
#define PIN_BUZZER          25      // active buzzer via NPN
#define PIN_MOTOR           26      // vibration motor via NPN / MOSFET
#define PIN_BATT_ADC        34      // ADC1_CH6, input only

// -----------------------------------------------------------------------------
// I2C addresses
// -----------------------------------------------------------------------------
#define ADDR_MPU6050        0x69    // AD0 tied HIGH
#define ADDR_DS3231         0x68
#define ADDR_EEPROM         0x50    // AT24C32 on the RTC module: A0-A2 bridged (default 0x57
                                    // would collide with the MAX30102)
#define ADDR_MAX30102       0x57

// -----------------------------------------------------------------------------
// Battery (1S LiPo through a 100k/100k divider)
// -----------------------------------------------------------------------------
#define BATT_DIVIDER_RATIO        2.0f
#define BATT_EMPTY_V              3.30f
#define BATT_FULL_V               4.20f
#define BATT_LOW_PCT              15
#define BATT_CHECK_PERIOD_S       30

// -----------------------------------------------------------------------------
// Motion / fall detection  (window constants must match ml_training/*.py)
// -----------------------------------------------------------------------------
#define MOTION_SAMPLE_HZ          50
#define FALL_WINDOW_SAMPLES       128           // 2.56 s
#define FALL_IMPACT_INDEX         64            // impact sample position inside the window
#define FALL_GYRO_SCALE_DPS       500.0f

#define FREEFALL_THRESHOLD_G      0.50f         // |a| below -> free fall
#define FREEFALL_MIN_SAMPLES      3             // >= 60 ms of free fall
#define IMPACT_THRESHOLD_G        2.70f         // |a| above -> impact (spec: 2.5-3.0 g)
#define IMPACT_WINDOW_MS          1000          // impact must follow free fall within 1 s
#define POST_IMPACT_SETTLE_MS     500           // ignore bounce right after impact
#define STILLNESS_MS              2500          // low motion required after impact
#define STILLNESS_ACC_DEV_G       0.15f         // max | |a| - 1 g | while "still"
#define STILLNESS_GYRO_DPS        40.0f         // max gyro magnitude while "still"
#define FALL_PROB_THRESHOLD       0.50f         // p(fall) from the CNN needed to start the countdown
#define FALL_USE_TFLM             1             // 0 = rule-based fallback score instead of the CNN
#define FALL_COUNTDOWN_S          15            // user can cancel with any button
#define FALL_COOLDOWN_MS          10000

// Motion gate for PPG (activity detection on |a| deviation and gyro magnitude)
#define MOTION_GATE_ACC_STD_G     0.06f
#define MOTION_GATE_GYRO_DPS      35.0f
#define MOTION_GATE_HOLD_MS       2000

// -----------------------------------------------------------------------------
// Vitals (MAX30102: 100 sps, 4x averaging -> 25 Hz into the algorithm)
// -----------------------------------------------------------------------------
#define PPG_SAMPLE_HZ             25
#define PPG_CONTACT_MIN_IR        30000         // below: no skin contact
#define RR_DECIMATION             5             // 25 Hz -> 5 Hz for the respiratory band
#define RR_BAND_LOW_HZ            0.10f
#define RR_BAND_HIGH_HZ           0.50f

#define HR_LOW_BPM                40
#define HR_HIGH_BPM               130
#define SPO2_LOW_PCT              90
#define RR_LOW_BPM                8
#define RR_HIGH_BPM               28
#define VITALS_ABNORMAL_PERSIST_S 30            // consecutive seconds before alerting
#define VITALS_ALERT_REARM_S      120

// -----------------------------------------------------------------------------
// Medication reminders (DS3231 alarm 1, daily). hour, minute, label
// -----------------------------------------------------------------------------
#define MED_SCHEDULE { {8, 0, "Metformin 500mg"}, {13, 0, "Amlodipine 5mg"}, {21, 0, "Atorvastatin 20mg"} }
#if SIMULATION_MODE
#define MED_CONFIRM_TIMEOUT_S     60            // shortened so the demo shows MED_MISSED quickly
#define SIM_MED_FIRST_S           30            // first simulated reminder after boot
#define SIM_MED_INTERVAL_S        150
#else
#define MED_CONFIRM_TIMEOUT_S     300           // spec: 5 minutes
#endif
#define MED_RENAG_PERIOD_S        60

// -----------------------------------------------------------------------------
// Buttons
// -----------------------------------------------------------------------------
#define BTN_DEBOUNCE_MS           40
#define BTN_LONG_PRESS_MS         2000

// -----------------------------------------------------------------------------
// FreeRTOS layout  (core 1 = sensing, core 0 = WiFi stack + comms)
// -----------------------------------------------------------------------------
#define MOTION_TASK_STACK         8192          // TFLM interpreter runs here
#define MOTION_TASK_PRIO          5
#define MOTION_TASK_CORE          1
#define VITALS_TASK_STACK         6144
#define VITALS_TASK_PRIO          4
#define VITALS_TASK_CORE          1
#define REMINDER_TASK_STACK       4096
#define REMINDER_TASK_PRIO        3
#define REMINDER_TASK_CORE        0
#define ALERT_TASK_STACK          8192
#define ALERT_TASK_PRIO           2
#define ALERT_TASK_CORE           0

#define ALERT_QUEUE_LEN_HIGH      8
#define ALERT_QUEUE_LEN_LOW       16
