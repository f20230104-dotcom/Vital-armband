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

**Live demo:** https://huggingface.co/spaces/kkshik/vital-wristband. It runs the simulation scenario, fall state machine,
alert log and a clickable wristband in the browser. The full Gradio dashboard (with the ESP32 webhook) runs locally with
`python dashboard/app.py --simulate`. The demo's source is [`web_demo/`](web_demo) (upload both files to a Static Space to redeploy).

```
 MPU6050 ─┐                    ┌─ motion_task  (50 Hz, fall FSM + TFLite Micro CNN) ─┐
 MAX30102 ┼─ hardware_hal ─────┼─ vitals_task  (PPG → HR, SpO₂, RR 0.1–0.5 Hz)        ├─▶ priority queues ─▶ alert_task ─▶ WiFi POST ─▶ server.py ─▶ app.py (Gradio)
 DS3231  ─┘  (or mock_sensors) └─ reminder_task (DS3231 alarm, 5-min confirm)        ─┘                     (WiFi on only while sending)
```

## Repository layout

| Path | What it is |
|---|---|
| `platformio.ini` | `esp32dev` (simulation, default), `esp32dev-hw` (real sensors), `esp32dev-logger` (raw wrist IMU logger), `native` (host unit tests) |
| `include/config.h` | Pin map, I²C addresses, thresholds, task layout, WiFi/webhook |
| `include/types.h` | Sample structs, `AlertMsg`, `VitalsSnapshot`, event-group bits, RTOS handles |
| `include/fall_model_data.h` | **Generated** int8 TFLite flatbuffer (13.8 KB) + quant params + boot self-test vectors |
| `include/mock_sensors.h` | Simulated IMU / PPG / battery driven by a scripted 240 s scenario |
| `src/main.cpp` | Queue/event-group setup, task creation, supervisor loop (SOS, battery, health) |
| `src/tasks/*.cpp` | `motion_task`, `vitals_task`, `reminder_task`, `alert_task` |
| `src/drivers/hardware_hal.cpp` | Real drivers (Adafruit MPU6050, SparkFun MAX3010x, RTClib, AT24C32) with per-sensor fallback to mocks |
| `src/drivers/dsp_filters.cpp` | Biquads, the 0.1–0.5 Hz respiratory band-pass, running stats |
| `src/drivers/ppg_processor.cpp` | HR (peak detection), SpO₂ (ratio-of-ratios), RR (baseline band-pass) with motion gate |
| `src/drivers/fall_detector.cpp` | Two-path trigger (free fall → impact, or impact > 2 g) → stillness → CNN → countdown |
| `src/ml/fall_classifier.cpp` | TensorFlow Lite Micro interpreter wrapper (6-op resolver, 24 KB arena) |
| `ml_training/` | `fetch_dataset.py` → `train_model.py` → `export_to_c.py`; `experiments.py`, `trigger_analysis.py`, `record_wrist_data.py`, `verify_sim_windows.py` |
| `dashboard/` | `server.py` (FastAPI webhook receiver + SQLite), `app.py` (Gradio UI), `device_simulator.py` |
| `test/test_core/` | 15 Unity tests for DSP, vitals and the fall state machine (`pio test -e native`) |
| `tools/host_sim.cpp` | Runs the firmware's sensing logic on a PC, faster than real time |
| `web_demo/` | Static in-browser command center deployed to the Hugging Face Space (`#t62` jumps to second 62) |
| `docs/` | Presentation (`.pptx`), its build script, dashboard screenshots |

## 1. ML pipeline (three datasets, 75 people)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml_training/requirements.txt
python ml_training/fetch_dataset.py             # SisFall (230 MB) + synthetic wrist windows -> data/dataset.npz
python ml_training/fetch_dataset.py --uci-only  # UCI wrist data (see below) -> data/uci_wrist.npz
python ml_training/train_model.py               # deployed recipe: + UCI wrist training + rotation augmentation
python ml_training/export_to_c.py               # -> include/fall_model_data.h
python ml_training/experiments.py               # evaluation study (~20 min) -> results/experiments.json + figures
python ml_training/trigger_analysis.py          # trigger recall study -> results/trigger.json
```

| Dataset | Sensor / placement | People | Trials → windows | Role |
|---|---|---|---|---|
| **SisFall** (Sucerquia et al., *Sensors* 2017) | ADXL345 + ITG3200, waist, 200 Hz | 38 (15 aged 60+) | 4,505 → 18,469 | train / validate / test |
| **UCI Simulated Falls & ADL** (Özdemir & Barshan, *Sensors* 2014) | Xsens MTw, **right wrist**, 25 Hz | 17 | 3,326 → 14,497 | 12 train, 5 unseen test |
| **Synthetic wrist** (our generator) | physics model of wrist falls / ADLs | 20 virtual | → 5,000 | augmentation |

* SisFall's official UdeA link is offline; `fetch_dataset.py` uses a GitHub mirror of the raw files and falls back to
  the synthetic generator. UCI dataset 455 is a 1.2 GB archive of nested RARs: extract only the right-wrist files
  (`*/340537.txt`, never the volunteers' `Bilgi.txt` info files) into `ml_training/data/raw/uci_falls/wrist/`.
* **Preparation:** unit conversion (g, °/s), resampling to 50 Hz, 128-sample windows with the impact at sample 64
  (±10 jitter, the same alignment the firmware uses), ADL hard negatives centred on each trial's 3 strongest peaks,
  **subject-wise** splits, augmentation (random 3-D rotation ±180°, gain 0.9–1.1, noise σ 0.02), balanced class weights.
* **Model:** 4×Conv(k,1) + GAP + Dense, 5,825 params, 182.8 k MACs, **13.8 KB int8**; ops CONV_2D, MAX_POOL_2D, MEAN,
  FULLY_CONNECTED, LOGISTIC, RESHAPE.
* Real recordings from your own wristband (`pio run -e esp32dev-logger` + `ml_training/record_wrist_data.py`) are
  picked up automatically from `ml_training/data/wrist/`.

**Deployed int8 model, 9,052 test windows from 17 people never used in training:**

| Test set | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SisFall (12 held-out subjects) | 98.1 % | 98.5 % | 97.1 % | 0.978 |
| **UCI real wrist (5 held-out volunteers)** | **98.4 %** | **99.1 %** | **98.4 %** | **0.987** |
| All | 98.3 % | 98.9 % | 97.9 % | 0.984 (AUC 0.995) |

**Experiments** (`ml_training/results/`, all subject-wise):

* **Baselines:** threshold rules F1 ≈ 0.25; rule-based score 0.85; Random Forest (42 features, ~1.4 MB) 0.98 SisFall /
  0.955 wrist; the 14 KB CNN 0.98 / 0.987.
* **Ablation:** real-wrist recall 80.2 % (SisFall only) → 97.3 % with rotation augmentation → 98.4 % with UCI wrist
  training + rotation (deployed).
* **5-fold cross-validation (75 people):** accuracy 98.9 ± 0.2 %, precision 99.3 ± 0.1 %, recall 98.7 ± 0.4 %,
  AUC 0.999.
* **Trigger analysis:** the free-fall (< 0.5 g, 60 ms) → impact (> 2.7 g) trigger alone fires for only 43–44 % of real
  falls, so the firmware adds an impact-only path (> 2.0 g, `IMPACT_ONLY_THRESHOLD_G`); trigger recall 99 %,
  system recall (trigger × CNN) 96.4 % SisFall / 97.8 % wrist.
* **Firmware simulation:** 80 simulated minutes: 20/20 falls, 0/40 false alerts from jumps and table slaps; vital-sign
  MAE HR 3.4 bpm, SpO₂ 1.1 %, RR 2.4 /min.

Presentation (BITS template, 34 slides): [`docs/Vital_Wristband_BITS_F312.pptx`](docs/Vital_Wristband_BITS_F312.pptx),
rebuilt from the results by `python docs/presentation/build_presentation.py`.

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
| 55 | forward fall, lying still | CNN p = 0.996 → 15 s countdown → **FALL** |
| 110 | lateral fall, button A at 120 s | **FALL_CANCELLED** |
| 132 | wrist slaps a table (4 g, no free fall) | nothing |
| 145–195 | hypoxia: SpO₂ 87 %, HR 122, RR 25 | **VITALS_ABNORMAL** after 30 s persistence |
| 225 | button A held 2 s | **SOS** |
| 30 + 150k | medication reminder | even doses confirmed → **MED_TAKEN**; odd → **MED_MISSED** (60 s in sim, 5 min on hardware) |

* Build with real sensors: `pio run -e esp32dev-hw`. Missing chips fall back to mocks individually
  (`AUTO_FALLBACK_TO_MOCK`); payloads carry `"sim": true` whenever any mock is active.
* **Hardware bring-up aids:** the hardware build prints an I²C bus scan against the expected addresses, runs the
  model self-test, logs inference time and tensor-arena use, and arms a task watchdog. Set the sensor mounting with
  `IMU_AXIS_SRC` / `IMU_AXIS_SIGN` and the wrist LED drive with `PPG_LED_AMPLITUDE` in `include/config.h`.
* **Collect real wrist data:** `pio run -e esp32dev-logger -t upload`, then
  `python ml_training/record_wrist_data.py --port <port> --subject W01 --activity F01 --seconds 15`
  (falls only by healthy volunteers onto crash mats).

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
pio test -e native                                    # 15 unit tests
c++ -std=c++17 -O2 -Iinclude tools/host_sim.cpp src/drivers/dsp_filters.cpp \
    src/drivers/ppg_processor.cpp src/drivers/fall_detector.cpp -o host_sim
./host_sim 480 windows.csv                            # firmware event timeline, 8 simulated minutes
python ml_training/verify_sim_windows.py windows.csv  # score firmware windows with the int8 model
```

> **Note:** PlatformIO cannot build from a folder whose name ends in a space; clone into a normal path.

## Disclaimer

Research and educational prototype, not a medical device. SpO₂ uses Maxim's empirical `110 − 25·R` line and must be
calibrated against a reference oximeter. The fall model is trained mostly on waist-worn data and simulated wrist data.
