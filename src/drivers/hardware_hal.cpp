// =============================================================================
// hardware_hal.cpp - Real sensor drivers with fallback to mock_sensors.h
// =============================================================================
#include "hardware_hal.h"

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <MAX30105.h>
#include <RTClib.h>
#include <freertos/timers.h>

#include "config.h"
#include "mock_sensors.h"

// ----------------------------------------------------------------------------- state
static bool s_sim[(int)Sensor::COUNT] = {true, true, true, true};
static char s_summary[96];

static Adafruit_MPU6050 s_mpu;
static MAX30105 s_max;
static RTC_DS3231 s_rtc;

static sim::MockImu s_mockImu;
static sim::MockPpg s_mockPpg((float)PPG_SAMPLE_HZ);
static uint32_t s_simPpgLastMs = 0;
static uint32_t s_unixAtBoot = 0;          // simulated RTC
static volatile bool s_rtcIrq = false;

// I2C guard ------------------------------------------------------------------------
namespace {
struct I2cLock {
    bool ok;
    I2cLock() : ok(g_i2cMutex == nullptr || xSemaphoreTake(g_i2cMutex, pdMS_TO_TICKS(50)) == pdTRUE) {}
    ~I2cLock() { if (ok && g_i2cMutex) xSemaphoreGive(g_i2cMutex); }
};
}

static void IRAM_ATTR rtcIsr() { s_rtcIrq = true; }

[[maybe_unused]] static bool i2cPresent(uint8_t addr) {
    I2cLock l;
    Wire.beginTransmission(addr);
    return Wire.endTransmission() == 0;
}

// ----------------------------------------------------------------------------- annunciator
static uint32_t s_alarmMask = 0;          // bit per AlarmPattern
static uint32_t s_alarmTick = 0;
static AlarmPattern s_alarmCurrent = AlarmPattern::NONE;
static TimerHandle_t s_alarmTimer = nullptr;
static portMUX_TYPE s_alarmMux = portMUX_INITIALIZER_UNLOCKED;

static AlarmPattern highestAlarm(uint32_t mask) {
    static const AlarmPattern order[] = {AlarmPattern::FALL_COUNTDOWN, AlarmPattern::SOS_ACK, AlarmPattern::ABNORMAL,
                                         AlarmPattern::MED_REMINDER, AlarmPattern::CONFIRM};
    for (AlarmPattern p : order)
        if (mask & (1u << (uint8_t)p)) return p;
    return AlarmPattern::NONE;
}

static void alarmTimerCb(TimerHandle_t) {          // 100 ms tick
    portENTER_CRITICAL(&s_alarmMux);
    AlarmPattern p = highestAlarm(s_alarmMask);
    if (p != s_alarmCurrent) { s_alarmCurrent = p; s_alarmTick = 0; }
    const uint32_t k = s_alarmTick++;
    portEXIT_CRITICAL(&s_alarmMux);

    bool buzz = false, vib = false;
    switch (p) {
        case AlarmPattern::FALL_COUNTDOWN: buzz = (k % 4) < 2; vib = (k % 10) < 6; break;          // urgent
        case AlarmPattern::SOS_ACK:        buzz = k < 10; vib = k < 10; if (k >= 10) hal_alarmStop(p); break;
        case AlarmPattern::ABNORMAL:       buzz = (k % 50) < 6 && (k % 2) == 0; vib = (k % 50) < 6; break;
        case AlarmPattern::MED_REMINDER:   buzz = (k % 20) == 0 || (k % 20) == 2; vib = (k % 20) < 5; break;
        case AlarmPattern::CONFIRM:        buzz = k < 1; if (k >= 1) hal_alarmStop(p); break;
        case AlarmPattern::NONE:           break;
    }
    digitalWrite(PIN_BUZZER, buzz ? HIGH : LOW);
    digitalWrite(PIN_MOTOR, vib ? HIGH : LOW);
}

void hal_alarmStart(AlarmPattern p) {
    if (p == AlarmPattern::NONE) return;
    portENTER_CRITICAL(&s_alarmMux);
    s_alarmMask |= 1u << (uint8_t)p;
    if (p == AlarmPattern::SOS_ACK || p == AlarmPattern::CONFIRM) s_alarmTick = 0;
    portEXIT_CRITICAL(&s_alarmMux);
}

void hal_alarmStop(AlarmPattern p) {
    portENTER_CRITICAL(&s_alarmMux);
    s_alarmMask &= ~(1u << (uint8_t)p);
    portEXIT_CRITICAL(&s_alarmMux);
}

// ----------------------------------------------------------------------------- init
[[maybe_unused]] static bool initImu() {
    I2cLock l;
    if (!s_mpu.begin(ADDR_MPU6050, &Wire)) return false;
    s_mpu.setAccelerometerRange(MPU6050_RANGE_16_G);      // impacts can exceed 8 g
    s_mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
    s_mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);         // anti-alias for 50 Hz sampling
    s_mpu.setSampleRateDivisor(19);                       // 1 kHz / (1 + 19) = 50 Hz
    // Motion interrupt on INT (GPIO27) - lets a future deep-sleep build wake on movement.
    s_mpu.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
    s_mpu.setMotionDetectionThreshold(2);
    s_mpu.setMotionDetectionDuration(20);
    s_mpu.setInterruptPinLatch(false);
    s_mpu.setInterruptPinPolarity(false);                 // active high
    s_mpu.setMotionInterrupt(true);
    return true;
}

[[maybe_unused]] static bool initPpg() {
    I2cLock l;
    if (!s_max.begin(Wire, I2C_CLOCK_HZ, ADDR_MAX30102)) return false;
    // LED PPG_LED_AMPLITUDE, 4x averaging, red+IR, 100 sps -> 25 Hz output, 411 us (18-bit), 16384 nA range
    s_max.setup(PPG_LED_AMPLITUDE, 4, 2, 100, 411, 16384);
    return true;
}

[[maybe_unused]] static bool initRtc() {
    {
        I2cLock l;
        if (!s_rtc.begin(&Wire)) return false;
        if (s_rtc.lostPower()) s_rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
        s_rtc.disable32K();
        s_rtc.writeSqwPinMode(DS3231_OFF);                 // INT/SQW pin used as alarm interrupt
        s_rtc.clearAlarm(1);
        s_rtc.clearAlarm(2);
        s_rtc.disableAlarm(2);
    }
    pinMode(PIN_RTC_INT, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_RTC_INT), rtcIsr, FALLING);
    return true;
}

bool hal_init() {
    pinMode(PIN_BUZZER, OUTPUT);
    pinMode(PIN_MOTOR, OUTPUT);
    digitalWrite(PIN_BUZZER, LOW);
    digitalWrite(PIN_MOTOR, LOW);
    pinMode(PIN_BTN_A, INPUT_PULLUP);
    pinMode(PIN_BTN_B, INPUT_PULLUP);
    pinMode(PIN_MPU_INT, INPUT);
    pinMode(PIN_MAX_INT, INPUT_PULLUP);
    analogReadResolution(12);
    analogSetPinAttenuation(PIN_BATT_ADC, ADC_11db);

    s_alarmTimer = xTimerCreate("alarm", pdMS_TO_TICKS(100), pdTRUE, nullptr, alarmTimerCb);
    if (s_alarmTimer) xTimerStart(s_alarmTimer, 0);

    s_unixAtBoot = DateTime(F(__DATE__), F(__TIME__)).unixtime();

#if SIMULATION_MODE
    for (bool& s : s_sim) s = true;
#else
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, I2C_CLOCK_HZ);
    // Bring-up aid: list every device on the bus next to what the wristband expects.
    Serial.print("[hal] I2C scan:");
    int found = 0;
    for (uint8_t a = 0x08; a < 0x78; ++a)
        if (i2cPresent(a)) { Serial.printf(" 0x%02X", a); ++found; }
    Serial.printf("  (%d devices; expect MPU6050 0x%02X, MAX30102 0x%02X, DS3231 0x%02X, EEPROM 0x%02X)\n", found,
                  ADDR_MPU6050, ADDR_MAX30102, ADDR_DS3231, ADDR_EEPROM);
    s_sim[(int)Sensor::IMU] = !(i2cPresent(ADDR_MPU6050) && initImu());
    s_sim[(int)Sensor::PPG] = !(i2cPresent(ADDR_MAX30102) && initPpg());
    s_sim[(int)Sensor::RTC] = !(i2cPresent(ADDR_DS3231) && initRtc());
    s_sim[(int)Sensor::EEPROM] = !i2cPresent(ADDR_EEPROM);
    if (!AUTO_FALLBACK_TO_MOCK && hal_anySimulated()) {
        Serial.printf("[hal] sensor missing and AUTO_FALLBACK_TO_MOCK=false: %s\n", hal_sensorSummary());
        return false;
    }
#endif
    s_simPpgLastMs = millis();
    Serial.printf("[hal] %s\n", hal_sensorSummary());
    return true;
}

bool hal_isSimulated(Sensor s) { return s_sim[(int)s]; }

bool hal_anySimulated() {
    for (bool s : s_sim) if (s) return true;
    return false;
}

const char* hal_sensorSummary() {
    snprintf(s_summary, sizeof(s_summary), "IMU=%s PPG=%s RTC=%s EEPROM=%s",
             s_sim[0] ? "sim" : "real", s_sim[1] ? "sim" : "real", s_sim[2] ? "sim" : "real",
             s_sim[3] ? "sim" : "real");
    return s_summary;
}

// ----------------------------------------------------------------------------- sensors
bool hal_readImu(ImuSample& out) {
    const uint32_t now = millis();
    if (s_sim[(int)Sensor::IMU]) {
        out = s_mockImu.sample(now);
        return true;
    }
    sensors_event_t a, g, temp;
    {
        I2cLock l;
        if (!l.ok || !s_mpu.getEvent(&a, &g, &temp)) return false;
    }
    constexpr float kG = 9.80665f, kDeg = 57.2957795f;
    static constexpr int kSrc[3] = IMU_AXIS_SRC;
    static constexpr int kSign[3] = IMU_AXIS_SIGN;
    const float acc[3] = {a.acceleration.x / kG, a.acceleration.y / kG, a.acceleration.z / kG};
    const float gyr[3] = {g.gyro.x * kDeg, g.gyro.y * kDeg, g.gyro.z * kDeg};
    out = ImuSample{kSign[0] * acc[kSrc[0]], kSign[1] * acc[kSrc[1]], kSign[2] * acc[kSrc[2]],
                    kSign[0] * gyr[kSrc[0]], kSign[1] * gyr[kSrc[1]], kSign[2] * gyr[kSrc[2]], now};
    return true;
}

int hal_readPpg(PpgSample* out, int max) {
    const uint32_t now = millis();
    int n = 0;
    if (s_sim[(int)Sensor::PPG]) {
        const uint32_t period = 1000 / PPG_SAMPLE_HZ;
        // "physical" motion from the scenario (not the detector's opinion) drives artefacts
        const sim::ImuMode m = sim::Scenario::mode(sim::Scenario::cycleTime(now));
        const bool moving = m == sim::ImuMode::WALK || m == sim::ImuMode::GESTURE;
        while (n < max && now - s_simPpgLastMs >= period) {
            s_simPpgLastMs += period;
            out[n++] = s_mockPpg.sample(s_simPpgLastMs, moving);
        }
        return n;
    }
    I2cLock l;
    if (!l.ok) return 0;
    s_max.check();
    while (n < max && s_max.available()) {
        out[n++] = PpgSample{s_max.getFIFORed(), s_max.getFIFOIR(), now};
        s_max.nextSample();
    }
    return n;
}

// ----------------------------------------------------------------------------- power
float hal_batteryVoltage() {
#if SIMULATION_MODE
    return sim::mockBatteryVoltage(millis());
#else
    uint32_t mv = 0;
    for (int i = 0; i < 16; ++i) mv += analogReadMilliVolts(PIN_BATT_ADC);
    return (mv / 16.0f) / 1000.0f * BATT_DIVIDER_RATIO;
#endif
}

uint8_t hal_batteryPercent(float v) {
    // piecewise-linear 1S LiPo discharge curve (rest voltage -> %)
    static const float V[] = {3.30f, 3.60f, 3.70f, 3.75f, 3.80f, 3.90f, 4.00f, 4.10f, 4.20f};
    static const float P[] = {0, 8, 20, 32, 45, 62, 77, 90, 100};
    if (v <= V[0]) return 0;
    for (int i = 1; i < 9; ++i)
        if (v <= V[i]) return (uint8_t)(P[i - 1] + (P[i] - P[i - 1]) * (v - V[i - 1]) / (V[i] - V[i - 1]));
    return 100;
}

// ----------------------------------------------------------------------------- RTC
uint32_t hal_rtcNow() {
    if (s_sim[(int)Sensor::RTC]) return s_unixAtBoot + millis() / 1000;
    I2cLock l;
    return l.ok ? s_rtc.now().unixtime() : s_unixAtBoot + millis() / 1000;
}

void hal_rtcFormat(uint32_t unix, char* buf, size_t n) {
    DateTime d(unix);
    snprintf(buf, n, "%04u-%02u-%02uT%02u:%02u:%02u", d.year(), d.month(), d.day(), d.hour(), d.minute(), d.second());
}

void hal_rtcHourMinute(uint32_t unix, uint8_t& hour, uint8_t& minute) {
    DateTime d(unix);
    hour = d.hour();
    minute = d.minute();
}

bool hal_rtcSetDailyAlarm(uint8_t hour, uint8_t minute) {
    if (s_sim[(int)Sensor::RTC]) return true;              // simulated schedule, see hal_rtcAlarmFired
    I2cLock l;
    if (!l.ok) return false;
    s_rtc.clearAlarm(1);
    return s_rtc.setAlarm1(DateTime(2000, 1, 1, hour, minute, 0), DS3231_A1_Hour);   // match hh:mm:ss daily
}

bool hal_rtcAlarmFired() {
#if SIMULATION_MODE
    // Accelerated schedule: first reminder after SIM_MED_FIRST_S, then every SIM_MED_INTERVAL_S.
    static uint32_t next_s = SIM_MED_FIRST_S;
    if (millis() / 1000 >= next_s) { next_s += SIM_MED_INTERVAL_S; return true; }
    return false;
#else
    if (s_sim[(int)Sensor::RTC]) return false;
    if (!s_rtcIrq) return false;
    s_rtcIrq = false;
    I2cLock l;
    if (l.ok && s_rtc.alarmFired(1)) { s_rtc.clearAlarm(1); return true; }
    return false;
#endif
}

// ----------------------------------------------------------------------------- buttons
void hal_pollButtons() {
    struct Btn { uint8_t pin; bool down; uint32_t since, lastChange; };
    static Btn b[2] = {{PIN_BTN_A, false, 0, 0}, {PIN_BTN_B, false, 0, 0}};
    const uint32_t now = millis();
    uint32_t bits = 0;
    for (int i = 0; i < 2; ++i) {
        const bool pressed = digitalRead(b[i].pin) == LOW;
        if (pressed == b[i].down || now - b[i].lastChange < BTN_DEBOUNCE_MS) continue;
        b[i].lastChange = now;
        b[i].down = pressed;
        if (pressed) { b[i].since = now; continue; }
        const uint32_t held = now - b[i].since;          // released: classify the press
        if (i == 0) bits |= held >= BTN_LONG_PRESS_MS ? EVT_BTN_A_LONG : EVT_BTN_A_SHORT;
        else bits |= EVT_BTN_B_SHORT;
    }
#if SIMULATION_MODE
    static uint32_t prev = 0;
    bits |= sim::Scenario::buttons(prev, now, SIM_MED_FIRST_S, SIM_MED_INTERVAL_S);
    prev = now;
#endif
    if (bits) xEventGroupSetBits(g_events, bits);
}

// ----------------------------------------------------------------------------- EEPROM log
// Layout: [0..1] next record index, records of 8 bytes from address 32 (page-aligned).
void hal_eepromLog(EventType e, uint32_t unix) {
    if (s_sim[(int)Sensor::EEPROM]) return;
    constexpr uint16_t kBase = 32, kRec = 8, kCount = (4096 - kBase) / kRec;
    I2cLock l;
    if (!l.ok) return;
    Wire.beginTransmission(ADDR_EEPROM);
    Wire.write(0); Wire.write(0);
    if (Wire.endTransmission(false) != 0) return;
    uint16_t idx = 0;
    if (Wire.requestFrom((int)ADDR_EEPROM, 2) == 2) idx = (Wire.read() << 8) | Wire.read();
    if (idx >= kCount) idx = 0;
    const uint16_t addr = kBase + idx * kRec;
    const uint8_t rec[kRec] = {(uint8_t)e, 0xA5, 0, 0, (uint8_t)(unix >> 24), (uint8_t)(unix >> 16),
                               (uint8_t)(unix >> 8), (uint8_t)unix};
    Wire.beginTransmission(ADDR_EEPROM);
    Wire.write(addr >> 8); Wire.write(addr & 0xFF);
    Wire.write(rec, kRec);
    Wire.endTransmission();
    delay(6);                                              // t_WR
    idx = (idx + 1) % kCount;
    Wire.beginTransmission(ADDR_EEPROM);
    Wire.write(0); Wire.write(0); Wire.write(idx >> 8); Wire.write(idx & 0xFF);
    Wire.endTransmission();
    delay(6);
}
