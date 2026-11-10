// =============================================================================
// hardware_hal.h - Hardware abstraction: real sensors or simulated ones
//
// Every task talks to the hardware only through these functions. When SIMULATION_MODE
// is true (or a chip is missing and AUTO_FALLBACK_TO_MOCK is true) the calls are served
// by the generators in mock_sensors.h, so the rest of the firmware cannot tell the
// difference. All I2C traffic is serialised with g_i2cMutex.
// =============================================================================
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "types.h"

enum class Sensor : uint8_t { IMU = 0, PPG, RTC, EEPROM, COUNT };

bool  hal_init();
bool  hal_isSimulated(Sensor s);
bool  hal_anySimulated();
const char* hal_sensorSummary();               // e.g. "IMU=real PPG=sim RTC=real EEPROM=real"

// ---- sensors -------------------------------------------------------------------
bool  hal_readImu(ImuSample& out);              // g, deg/s
int   hal_readPpg(PpgSample* out, int max);     // drains the FIFO, returns #samples (25 Hz)

// ---- power ---------------------------------------------------------------------
float   hal_batteryVoltage();
uint8_t hal_batteryPercent(float volts);

// ---- RTC (DS3231) --------------------------------------------------------------
uint32_t hal_rtcNow();                          // unix seconds
void     hal_rtcFormat(uint32_t unix, char* buf, size_t n);   // ISO-8601 "YYYY-MM-DDTHH:MM:SS"
void     hal_rtcHourMinute(uint32_t unix, uint8_t& hour, uint8_t& minute);
bool     hal_rtcSetDailyAlarm(uint8_t hour, uint8_t minute);
bool     hal_rtcAlarmFired();                   // true once per alarm (clears it)

// ---- buttons / annunciator -----------------------------------------------------
void  hal_pollButtons();                        // call every ~20 ms; sets EVT_BTN_* bits
void  hal_alarmStart(AlarmPattern p);           // buzzer + vibration patterns, highest wins
void  hal_alarmStop(AlarmPattern p);

// ---- non-volatile event log (AT24C32 @ 0x50) -------------------------------------
void  hal_eepromLog(EventType e, uint32_t unix);
