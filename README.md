---
title: Vital Wristband Command Center
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: dashboard/app.py
short_description: Fall detection & vitals dashboard for an ESP32 wristband
pinned: false
---

# Vital Wristband

ESP32 smart-health wristband: **TinyML fall detection**, **heart rate / SpO₂ / respiratory rate**,
**medication reminders**, and a **live caregiver dashboard**. The whole stack runs today without any hardware:
the firmware has a full `SIMULATION_MODE`, and the dashboard ships with a virtual wristband.

**Live demo:** https://claude.ai/artifact/BFwt3ebfJPpfAAfy6G1xHP. It runs the simulation scenario, fall state machine,
alert log and a clickable wristband in the browser. The full Gradio dashboard runs locally with `python dashboard/app.py --simulate`.

```
 MPU6050 ─┐                    ┌─ motion_task  (50 Hz, fall FSM + TFLite Micro CNN) ─┐
 MAX30102 ┼─ hardware_hal ─────┼─ vitals_task  (PPG → HR, SpO₂, RR 0.1–0.5 Hz)        ├─▶ priority queues ─▶ alert_task ─▶ WiFi POST ─▶ server.py ─▶ app.py (Gradio)
 DS3231  ─┘  (or mock_sensors) └─ reminder_task (DS3231 alarm, 5-min confirm)        ─┘                     (WiFi on only while sending)
```

## Repository layout

| Path | What it is |
|---|---|
| `platformio.ini` | `esp32dev` (simulation, default), `esp32dev-hw` (real sensors), `native` (host unit tests) |
| `include/config.h` | Pin map, I²C addresses, thresholds, task layout, WiFi/webhook |
| `include/types.h` | Sample structs, `AlertMsg`, `VitalsSnapshot`, event-group bits, RTOS handles |
| `include/fall_model_data.h` | **Generated** int8 TFLite flatbuffer (14 KB) + quant params + boot self-test vectors |
| `include/mock_sensors.h` | Simulated IMU / PPG / battery driven by a scripted 240 s scenario |
| `src/main.cpp` | Queue/event-group setup, task creation, supervisor loop (SOS, battery, health) |
| `src/tasks/*.cpp` | `motion_task`, `vitals_task`, `reminder_task`, `alert_task` |
| `src/drivers/hardware_hal.cpp` | Real drivers (Adafruit MPU6050, SparkFun MAX3010x, RTClib, AT24C32) with per-sensor fallback to mocks |
| `src/drivers/dsp_filters.cpp` | Biquads, the 0.1–0.5 Hz respiratory band-pass, running stats |
| `src/drivers/ppg_processor.cpp` | HR (peak detection), SpO₂ (ratio-of-ratios), RR (baseline band-pass) with motion gate |
| `src/drivers/fall_detector.cpp` | `IDLE → FREE_FALL → IMPACT → POST_IMPACT → INFERENCE → COUNTDOWN` state machine |
| `src/ml/fall_classifier.cpp` | TensorFlow Lite Micro interpreter wrapper (6-op resolver, 24 KB arena) |
| `ml_training/` | `fetch_dataset.py` → `train_model.py` → `export_to_c.py` (+ `verify_sim_windows.py`) |
| `dashboard/` | `server.py` (FastAPI webhook receiver + SQLite), `app.py` (Gradio UI), `device_simulator.py` |
| `test/test_core/` | 14 Unity tests for DSP, vitals and the fall state machine (`pio test -e native`) |
| `tools/host_sim.cpp` | Runs the firmware's sensing logic on a PC, faster than real time |

## 1. ML pipeline (official SisFall dataset)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml_training/requirements.txt
python ml_training/fetch_dataset.py   # downloads SisFall (230 MB), windows it, adds synthetic wrist data
python ml_training/train_model.py     # 1D-CNN, subject-wise split, int8 quantisation
python ml_training/export_to_c.py     # -> include/fall_model_data.h
```

* **Dataset:** SisFall (Sucerquia et al., *Sensors* 2017): 38 subjects incl. 15 elderly, 19 ADL + 15 fall types,
  4,505 trials. The official UdeA link is offline, so the script falls back to a GitHub mirror of the raw files,
  and finally to a fully synthetic generator (free-fall valley < 0.5 g, impact 2.5–7 g, post-impact stillness).
* **Windows:** 200 Hz → 50 Hz, 128 samples (2.56 s), 6 channels (accel g, gyro dps/500). Falls are centred on the
  impact peak (±10 samples jitter); ADL negatives are centred on each trial's strongest peaks: the hard cases
  the threshold trigger would actually send to the model.
* **SisFall is waist-worn**, so 5,000 synthetic *wrist* windows (arm swing, jumps, table slaps, sit-downs,
  falls with orientation change) are mixed in. Re-train on real wrist data once you have the hardware.
* **Model:** 4×Conv(k,1) + GAP + Dense, 5,825 params, **14 KB int8**, ops: CONV_2D, MAX_POOL_2D, MEAN,
  FULLY_CONNECTED, LOGISTIC, RESHAPE.

**Held-out-subject results (int8 model, 4,767 test windows from 12 unseen subjects):**

| | Accuracy | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|
| All | 98.83 % | 99.47 % | 97.88 % | 98.67 % | 99.58 % |
| SisFall only | 98.78 % | 99.51 % | 97.71 % | 98.60 % | 99.62 % |
| Synthetic wrist | 99.20 % | 99.20 % | 99.20 % | 99.20 % | 99.20 % |

float32 vs int8 decision agreement: 99.87 %.

## 2. Dashboard

```bash
pip install -r requirements.txt
python dashboard/app.py --simulate          # UI http://localhost:7860, receiver :8000, virtual wristband
python dashboard/app.py --simulate --share  # + public *.gradio.live link
python dashboard/server.py --port 8000      # receiver only (what the ESP32 posts to)
python dashboard/app.py --combined          # UI + webhook on ONE port (Docker / Render / HF Spaces)
```

Webhook: `POST /api/telemetry` (alias `/webhook`):

```json
{"device_id": "WB-01", "event": "FALL", "priority": 3, "hr": 112, "spo2": 95, "rr": 18,
 "battery_v": 3.82, "battery_pct": 71, "fall_prob": 0.97, "peak_g": 4.3, "detail": "...", "seq": 41}
```

Events: `TELEMETRY FALL FALL_CANCELLED SOS MED_REMINDER MED_TAKEN MED_MISSED VITALS_ABNORMAL LOW_BATTERY BOOT`.
Critical ones (`FALL SOS MED_MISSED VITALS_ABNORMAL`) flash in the banner and alert log until a caregiver
acknowledges them. Optional shared secret: set `WRISTBAND_API_KEY` on the server and `WEBHOOK_API_KEY` in the firmware.

**Deploy permanently on Hugging Face Spaces:** this README's front-matter already declares a Gradio Space
(`app_file: dashboard/app.py`). On a Space the app automatically runs in combined + simulate mode.

## 3. Firmware

```bash
cp include/secrets.example.h include/secrets.h   # WiFi + your PC's LAN IP
pio run -e esp32dev -t upload && pio device monitor
```

* Without `secrets.h` the alert task runs in **dry-run** mode and prints every JSON payload to Serial.
* `SIMULATION_MODE=true` (default env) replays this scenario every 240 s:

| t (s) | Scenario | Expected firmware output |
|---|---|---|
| 15–40 | walking | PPG motion-gated (values frozen, no false alarms) |
| 35 | jump (real free fall + 3.8 g landing) | `FREE_FALL → IMPACT → IDLE (wearer recovered)` |
| 55 | forward fall, lying still | CNN p≈0.996 → 15 s countdown → **FALL** |
| 110 | lateral fall, button A at 120 s | **FALL_CANCELLED** |
| 132 | wrist slaps a table (4 g, no free fall) | nothing |
| 145–195 | hypoxia: SpO₂ 87 %, HR 122, RR 25 | **VITALS_ABNORMAL** after 30 s persistence |
| 225 | button A held 2 s | **SOS** |
| 30 + 150k | medication reminder | even doses confirmed → **MED_TAKEN**; odd → **MED_MISSED** (60 s in sim, 5 min on hardware) |

* Build with real sensors: `pio run -e esp32dev-hw`. Missing chips fall back to mocks individually
  (`AUTO_FALLBACK_TO_MOCK`); payloads carry `"sim": true` whenever any mock is active.

### Pin map

| Function | GPIO | I²C addr |
|---|---|---|
| I²C SDA / SCL | 21 / 22 | |
| MPU6050 INT | 27 | 0x69 (AD0 high) |
| DS3231 INT/SQW | 33 | 0x68, AT24C32 EEPROM 0x50 (A0–A2 bridged; default 0x57 clashes with the MAX30102) |
| MAX30102 INT | 32 | 0x57 |
| Button A (cancel / SOS long-press) | 4 | |
| Button B (confirm medication) | 13 | |
| Buzzer / vibration motor | 25 / 26 | |
| Battery ADC (100k/100k divider) | 34 | |

### RTOS layout

| Task | Core | Prio | Period | Notes |
|---|---|---|---|---|
| motion | 1 | 5 | 20 ms | fall FSM, TFLM inference (~ms), owns `EVT_MOTION_ACTIVE` |
| vitals | 1 | 4 | 40 ms | drains MAX30102 FIFO, 1 s estimates → mailbox queue |
| reminder | 0 | 3 | 250 ms | DS3231 alarm 1 via INT pin, nag/confirm/timeout |
| alert | 0 | 2 | event | queue set (HIGH first), WiFi on → POST → WiFi off, retry buffer |
| loop | 1 | 1 | 20 ms | buttons, SOS, battery, stack/heap stats |

## 4. Verification without hardware

```bash
pio test -e native                                    # 14 unit tests
c++ -std=c++17 -O2 -Iinclude tools/host_sim.cpp src/drivers/dsp_filters.cpp \
    src/drivers/ppg_processor.cpp src/drivers/fall_detector.cpp -o host_sim
./host_sim 480 windows.csv                            # firmware event timeline, 8 simulated minutes
python ml_training/verify_sim_windows.py windows.csv  # score firmware windows with the int8 model
```

> **Note:** PlatformIO cannot build from a folder whose name ends in a space; clone into a normal path.

## Disclaimer

Research and educational prototype, not a medical device. SpO₂ uses Maxim's empirical `110 − 25·R` line and must be
calibrated against a reference oximeter. The fall model is trained mostly on waist-worn data and simulated wrist data.
