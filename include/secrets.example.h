// Copy to include/secrets.h (git-ignored) and fill in. config.h picks it up automatically.
#pragma once
#define WIFI_SSID        "my-home-wifi"
#define WIFI_PASSWORD    "my-wifi-password"
// LAN IP of the computer running `python dashboard/app.py` (or server.py)
#define WEBHOOK_URL      "http://192.168.1.50:8000/api/telemetry"
#define WEBHOOK_API_KEY  ""   // same value as the server's WRISTBAND_API_KEY env var, or empty
