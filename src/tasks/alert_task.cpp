// =============================================================================
// alert_task.cpp - Priority queue listener and WiFi webhook dispatcher
//
//  * Two queues (HIGH: fall/SOS/missed meds/abnormal vitals, LOW: info) joined in a
//    FreeRTOS queue set; HIGH messages are always drained first.
//  * Messages >= IMMEDIATE_PRIORITY open a WiFi window right away; LOW ones ride along
//    with the next TELEMETRY heartbeat.
//  * WiFi is powered up only for the POSTs and switched off afterwards (WIFI_OFF).
//  * Undelivered HIGH/CRITICAL messages are kept in a retry buffer (retried every 30 s).
// =============================================================================
#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <math.h>
#include <string.h>

#include "config.h"
#include "types.h"
#include "hardware_hal.h"

namespace {

constexpr int kBatchMax = 24;
constexpr uint32_t kRetryPeriodMs = 30000;

AlertMsg s_retry[RETRY_BUFFER_LEN];
int s_retryN = 0;
AlertMsg s_low[ALERT_QUEUE_LEN_LOW];
int s_lowN = 0;
uint32_t s_seq = 0;

bool credentialsConfigured() { return strcmp(WIFI_SSID, "YOUR_WIFI_SSID") != 0 && strlen(WIFI_SSID) > 0; }

bool wifiUp() {
    if (WiFi.status() == WL_CONNECTED) return true;
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    const uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_CONNECT_TIMEOUT_MS) vTaskDelay(pdMS_TO_TICKS(100));
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[alert] WiFi up in %lu ms, IP %s, RSSI %d\n", (unsigned long)(millis() - t0),
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
        return true;
    }
    Serial.println("[alert] WiFi connect timeout");
    return false;
}

void wifiDown() {
    WiFi.disconnect(true, false);
    WiFi.mode(WIFI_OFF);                       // radio off between windows (~100 mA saved)
}

size_t buildJson(const AlertMsg& m, char* out, size_t n) {
    JsonDocument doc;
    doc["device_id"] = DEVICE_ID;
    doc["event"] = eventName(m.type);
    doc["priority"] = m.priority;
    VitalsSnapshot v;
    if (xQueuePeek(g_vitalsMailbox, &v, 0) == pdTRUE) {
        if (!isnan(v.hr)) doc["hr"] = lroundf(v.hr);
        if (!isnan(v.spo2)) doc["spo2"] = roundf(v.spo2 * 10) / 10;
        if (!isnan(v.rr)) doc["rr"] = lroundf(v.rr);
        doc["motion"] = v.motion_gated;
    }
    const float bv = hal_batteryVoltage();
    doc["battery_v"] = roundf(bv * 100) / 100;
    doc["battery_pct"] = hal_batteryPercent(bv);
    if (!isnan(m.fall_prob)) doc["fall_prob"] = roundf(m.fall_prob * 1000) / 1000;
    if (!isnan(m.peak_g)) doc["peak_g"] = roundf(m.peak_g * 100) / 100;
    if (m.detail[0]) doc["detail"] = m.detail;
    doc["seq"] = ++s_seq;
    doc["uptime_s"] = millis() / 1000;
    char ts[24];
    hal_rtcFormat(m.unix_time ? m.unix_time : hal_rtcNow(), ts, sizeof(ts));
    doc["ts"] = ts;
    doc["sim"] = hal_anySimulated();
    return serializeJson(doc, out, n);
}

bool post(const char* body, size_t len) {
    for (int attempt = 0; attempt < HTTP_MAX_RETRIES; ++attempt) {
        HTTPClient http;
        http.setTimeout(HTTP_TIMEOUT_MS);
        http.setConnectTimeout(HTTP_TIMEOUT_MS);
        if (http.begin(WEBHOOK_URL)) {
            http.addHeader("Content-Type", "application/json");
            if (strlen(WEBHOOK_API_KEY)) http.addHeader("X-API-Key", WEBHOOK_API_KEY);
            const int code = http.POST((uint8_t*)body, len);
            http.end();
            if (code >= 200 && code < 300) return true;
            Serial.printf("[alert] POST failed (%d) attempt %d\n", code, attempt + 1);
        }
        vTaskDelay(pdMS_TO_TICKS(250 << attempt));       // 250, 500, 1000 ms backoff
    }
    return false;
}

void keepForRetry(const AlertMsg& m) {
    if (m.priority < PRIO_HIGH) {
        Serial.printf("[alert] dropping undelivered %s (low priority)\n", eventName(m.type));
        return;
    }
    if (s_retryN < RETRY_BUFFER_LEN) s_retry[s_retryN++] = m;
    else Serial.printf("[alert] retry buffer full, dropping %s\n", eventName(m.type));
}

// Receive one message from the queue set, HIGH queue first. Every select must be
// matched by exactly one receive so the set never holds stale handles.
bool receiveOne(AlertMsg& m, TickType_t wait) {
    if (xQueueSelectFromSet(g_alertQueueSet, wait) == nullptr) return false;
    if (xQueueReceive(g_alertQueueHigh, &m, 0) == pdTRUE) return true;
    return xQueueReceive(g_alertQueueLow, &m, 0) == pdTRUE;
}

}  // namespace

void alertTask(void*) {
    uint32_t lastTelemetry = millis() - (TELEMETRY_PERIOD_S - 5) * 1000UL;   // first heartbeat ~5 s after boot
    uint32_t lastRetry = 0;
    static AlertMsg batch[kBatchMax];
    static char json[512];

    Serial.printf("[alert] webhook %s, WiFi \"%s\"%s\n", WEBHOOK_URL, WIFI_SSID,
                  credentialsConfigured() ? "" : "  (not configured -> DRY RUN: JSON printed to Serial)");

    for (;;) {
        const uint32_t sinceHb = millis() - lastTelemetry;
        uint32_t waitMs = sinceHb >= TELEMETRY_PERIOD_S * 1000UL ? 0 : TELEMETRY_PERIOD_S * 1000UL - sinceHb;
        if (s_retryN && waitMs > kRetryPeriodMs) waitMs = kRetryPeriodMs;

        int n = 0;
        AlertMsg m;
        bool immediate = false;
        if (receiveOne(m, pdMS_TO_TICKS(waitMs))) {
            do {
                if (m.priority >= IMMEDIATE_PRIORITY) { batch[n++] = m; immediate = true; }
                else if (s_lowN < ALERT_QUEUE_LEN_LOW) s_low[s_lowN++] = m;
                else batch[n++] = m;                                  // low buffer full: flush now
            } while (n < kBatchMax - RETRY_BUFFER_LEN - 1 && receiveOne(m, 0));
        }
        const bool heartbeatDue = millis() - lastTelemetry >= TELEMETRY_PERIOD_S * 1000UL;
        const bool retryDue = s_retryN && millis() - lastRetry >= kRetryPeriodMs;
        if (!immediate && !heartbeatDue && !retryDue) continue;

        // assemble: retries + new HIGH first, then low-priority backlog, then the heartbeat
        for (int i = 0; i < s_retryN && n < kBatchMax; ++i) batch[n++] = s_retry[i];
        s_retryN = 0;
        lastRetry = millis();
        for (int i = 0; i < s_lowN && n < kBatchMax - 1; ++i) batch[n++] = s_low[i];
        s_lowN = 0;
        if (heartbeatDue) {
            AlertMsg hb{};
            hb.type = EventType::TELEMETRY;
            hb.priority = PRIO_LOW;
            hb.t_ms = millis();
            hb.unix_time = hal_rtcNow();
            hb.fall_prob = hb.peak_g = NAN;
            batch[n++] = hb;
            lastTelemetry = millis();
        }
        // stable sort by priority (descending)
        for (int i = 1; i < n; ++i) {
            AlertMsg k = batch[i];
            int j = i - 1;
            while (j >= 0 && batch[j].priority < k.priority) { batch[j + 1] = batch[j]; --j; }
            batch[j + 1] = k;
        }

        if (!credentialsConfigured()) {
            for (int i = 0; i < n; ++i) {
                const size_t len = buildJson(batch[i], json, sizeof(json));
                Serial.printf("[alert] DRY-RUN POST %s %.*s\n", WEBHOOK_URL, (int)len, json);
            }
            continue;
        }

        const uint32_t t0 = millis();
        if (!wifiUp()) {
            for (int i = 0; i < n; ++i) keepForRetry(batch[i]);
            wifiDown();
            continue;
        }
        int ok = 0;
        for (int i = 0; i < n; ++i) {
            const size_t len = buildJson(batch[i], json, sizeof(json));
            if (post(json, len)) {
                ++ok;
                if (batch[i].type != EventType::TELEMETRY) Serial.printf("[alert] sent %s\n", eventName(batch[i].type));
            } else {
                keepForRetry(batch[i]);
            }
        }
        wifiDown();
        Serial.printf("[alert] %d/%d delivered, radio on %lu ms\n", ok, n, (unsigned long)(millis() - t0));
    }
}
