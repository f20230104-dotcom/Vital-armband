#!/usr/bin/env python3
"""
make_case_pcb.py - CONCEPT: the Vital wristband pod with every module replaced by one custom PCB.

    python hardware/enclosure/make_case_pcb.py

Phase-2 / future-work model, not a buildable design yet: part sizes are datasheet values for the parts
proposed below, and the PCB itself has not been laid out. It writes
  case_base_pcb.stl, case_lid_pcb.stl   printable concept case (43 x 33 x 10.3 mm)
  assembly_pcb.glb                      case + all parts, assembled, in colour
  viewer/pcb_parts.js + case meshes     the "Custom PCB" version of the 3D viewer
and runs the same fit check as assembly_xiao.py (exit 1 if anything collides).

Stack (skin up): 1.4 mm floor with the optical sensor flush in a window | 1.0 mm PCB |
2.5 mm of top-side parts (ESP32-S3-MINI-1, IMU, charger, ERM, buzzer, side buttons) |
4 mm LiPo 402530 | 0.1 mm gap | 1.2 mm lid.
Charging and flashing go through 4 pogo pads on the skin side (5 V, GND, USB D+/D-): no side opening.
Power on/off is a long press on button A (load switch + ESP32 deep sleep), so there is no slide switch.
"""
from __future__ import annotations

import base64
import json
import re
import sys

import numpy as np
from manifold3d import Manifold

from assembly_xiao import EPS, write_glb
from make_case import OUT, box, cyl_y, rrect, stl_bytes, write_stl

P = dict(
    inner_l=40.0, inner_w=30.0, inner_r=1.5, wall=1.5, floor=1.4, lid_top=1.2, chamfer=1.2,
    pcb_t=1.0, parts_h=2.6, battery_h=4.0, gap_h=0.1,
    rim_h=1.0, rim_t=0.8, fit=0.2,
    ppg_x=-4.0, ppg_win=(5.5, 7.0),                  # MAXM86161 window in the floor
    pogo_x=14.0, pogo_y=(-4.5, -1.5, 1.5, 4.5), pogo_d=2.4,
    btn_x=(-17.0, 16.5), btn_d=2.2,                  # side-push switches, +Y wall
    buzz_xy=(14.5, 6.0), buzz_hole=1.2,
    strap_w=22.0, bar_out=5.0, bar_gap=2.8, bar_d=2.5, bar_cheek=2.5, bar_h=4.5,
)
F = P["floor"]
PCB0, PCB1 = F, F + P["pcb_t"]                       # 1.4 .. 2.4
HB = PCB1 + P["parts_h"]                             # base top 5.0
B0, B1 = HB, HB + P["battery_h"]                     # battery 5.0 .. 9.0
CEIL = B1 + P["gap_h"]                               # 9.1


def cyl_z(cx, cy, r, z0, z1, seg=40):
    return Manifold.cylinder(z1 - z0, r, circular_segments=seg).translate([cx, cy, z0])


def build(p=P):
    L, W = p["inner_l"] + 2 * p["wall"], p["inner_w"] + 2 * p["wall"]
    R, c = p["inner_r"] + p["wall"], p["chamfer"]
    shell = Manifold.batch_hull([rrect(L - 2 * c, W - 2 * c, R - c).extrude(0.01), rrect(L, W, R).extrude(c)]) + \
        rrect(L, W, R).extrude(HB - c).translate([0, 0, c])
    base = shell - rrect(p["inner_l"], p["inner_w"], p["inner_r"]).extrude(HB).translate([0, 0, F])
    base -= rrect(*p["ppg_win"], 1.0).extrude(F + 2).translate([p["ppg_x"], 0, -1])
    for y in p["pogo_y"]:
        base -= cyl_z(p["pogo_x"], y, p["pogo_d"] / 2, -1, F + 1, 24)
    zb = PCB1 + 0.7                                                    # side-switch actuator height
    for x in p["btn_x"]:
        base -= cyl_y(x, zb, p["btn_d"], p["inner_w"] / 2 - 1, W / 2 + 1)
    sw, out, gap = p["strap_w"] + 1.0, p["bar_out"], p["bar_gap"]
    for s in (1, -1):
        y0, y1 = sorted((s * (W / 2 - 0.5), s * (W / 2 + out)))
        for xs in (-1, 1):
            base += box(*sorted((xs * sw / 2, xs * (sw / 2 + p["bar_cheek"]))), y0, y1, 0, p["bar_h"])
        bar = Manifold.cylinder(sw + 1, p["bar_d"] / 2, circular_segments=40).rotate([0, 90, 0])
        base += bar.translate([-sw / 2 - 0.5, s * (W / 2 + gap + p["bar_d"] / 2), p["bar_h"] / 2])

    ri_l, ri_w = p["inner_l"] - 2 * p["fit"], p["inner_w"] - 2 * p["fit"]
    li_l, li_w = ri_l - 2 * p["rim_t"], ri_w - 2 * p["rim_t"]
    r_in = max(0.5, p["inner_r"] - p["fit"] - p["rim_t"])
    ht = CEIL - HB
    lid = rrect(L, W, R).extrude(ht + p["lid_top"]).translate([0, 0, HB]) - \
        rrect(li_l, li_w, r_in).extrude(ht + 0.01).translate([0, 0, HB - 0.01])
    rim = rrect(ri_l, ri_w, p["inner_r"] - p["fit"]).extrude(p["rim_h"] + 0.01) - \
        rrect(li_l, li_w, r_in).extrude(p["rim_h"] + 1).translate([0, 0, -0.5])
    lid += rim.translate([0, 0, HB - p["rim_h"]])
    bx_, by_ = p["buzz_xy"]
    for dx, dy in ((-1.8, 0), (0, 0), (1.8, 0)):
        lid -= cyl_z(bx_ + dx, by_ + dy, p["buzz_hole"] / 2, CEIL - 1, CEIL + p["lid_top"] + 1, 20)
    return base, lid


# ----------------------------------------------------------------------------- parts
# id, name, short label, layer, explode offset, display size, info..., pieces [(solid, "#rrggbb", metal)]
GOLD, SILVER = "#d4a64a", "#c7ccd3"
zb = PCB1 + 0.7
PARTS = [
    dict(id="p-pcb", name="Custom PCB (4-layer)", short="Custom PCB", layer="mid", ex=8, dim="39 × 29 × 1.0",
         role="One board replaces the XIAO, the three sensor breakouts, the charger and the driver strip.",
         size="39 × 29 × 1.0 mm, 4-layer FR-4", where="Rests on the floor; sensor and pogo pads on the skin side, everything else on top",
         links="All parts below are soldered to it; the LiPo is soldered to two pads",
         note="Design in KiCad, then order assembled from JLCPCB or PCBWay (about 1–2 weeks).",
         pieces=[(box(-19.5, 19.5, -14.5, 14.5, PCB0, PCB1), "#1f5f3a", 0.0),
                 (box(-3.6, -2.0, -11.5, -9.5, PCB1, PCB1 + 0.5), "#111111", 0.0),          # decoupling caps lumped
                 (box(-3.6, -2.4, 9.4, 10.6, PCB1, PCB1 + 0.6), SILVER, 0.6)]),            # 32 kHz crystal
    dict(id="p-mcu", name="ESP32-S3-MINI-1 module", short="ESP32-S3-MINI-1", layer="mid", ex=8, dim="15.4 × 20.5",
         role="Processor and WiFi with the antenna built in: same ESP32-S3 and same firmware as the XIAO build.",
         size="15.4 × 20.5 × 2.4 mm", where="Top side, antenna end at the +X edge where the battery does not cover it",
         links="Native USB to the pogo pads for flashing; I²C to the IMU and PPG sensor",
         note="Keep copper and the battery away from the antenna end (the last ~5 mm).",
         pieces=[(box(-1.7, 18.8, -13.7, 1.7, PCB1, PCB1 + 0.8), "#1b2230", 0.0),
                 (box(-1.4, 13.6, -13.4, 1.4, PCB1 + 0.8, PCB1 + 2.4), SILVER, 0.8),
                 (box(14.2, 18.4, -12.0, 0.0, PCB1 + 0.8, PCB1 + 0.85), GOLD, 0.7)]),
    dict(id="p-imu", name="IMU BMI270", short="BMI270", layer="mid", ex=8, dim="2.5 × 3.0",
         role="Accelerometer + gyroscope for fall detection; replaces the end-of-life MPU6050 and can wake the ESP32 on motion.",
         size="2.5 × 3.0 × 0.83 mm LGA", where="Top side, near the centre of the board",
         links="I²C + interrupt to the ESP32-S3",
         note="Needs a new driver; record a little data with the logger build to confirm the model accuracy.",
         pieces=[(box(-7.25, -4.75, -9.5, -6.5, PCB1, PCB1 + 0.83), "#111111", 0.0)]),
    dict(id="p-charger", name="Charger + power path", short="Charger", layer="mid", ex=8, dim="3 × 3",
         role="Charges the LiPo from the pogo pads and switches the power on and off.",
         size="MCP73831 (SOT-23-5) + load switch + 3.3 V LDO", where="Top side, −X end",
         links="5 V from the pogo pads; battery pads; 3.3 V rail",
         note="Set the charge current to about 0.5C (100 mA for a 200 mAh cell).",
         pieces=[(box(-15.5, -12.5, -11.5, -8.5, PCB1, PCB1 + 1.1), "#111111", 0.0),
                 (box(-15.0, -13.0, -7.0, -5.0, PCB1, PCB1 + 0.6), "#111111", 0.0),
                 (box(-10.5, -7.6, -12.5, -10.1, PCB1, PCB1 + 1.1), "#111111", 0.0)]),
    dict(id="p-motor", name="Coin vibration motor 0825", short="Vibration motor", layer="mid", ex=8, dim="Ø8 × 2.5",
         role="Silent alert on the wrist.", size="8 mm × 2.5 mm", where="Top side, under the battery",
         links="MOSFET driver on the board", note="An LRA with a DRV2605L driver gives crisper, quieter haptics.",
         pieces=[(cyl_z(-11.0, 6.0, 4.0, PCB1, PCB1 + 2.5), SILVER, 0.8)]),
    dict(id="p-buzzer", name="SMD buzzer 5 × 5 mm", short="Buzzer", layer="mid", ex=8, dim="5 × 5 × 2",
         role="Audible alarm for falls, low SpO₂ and missed doses.", size="5 × 5 × 2 mm magnetic buzzer",
         where="Top side, +X end, under the sound holes (outside the battery)", links="MOSFET driver on the board",
         note="Quieter than the 9 mm buzzer; the vibration motor is the main alert.",
         pieces=[(box(12.0, 17.0, 3.5, 8.5, PCB1, PCB1 + 2.0), "#26282c", 0.0)]),
    dict(id="p-buttons", name="Side-push buttons ×2", short="Buttons", layer="mid", ex=8, dim="3.5 × 2.9",
         role="A: cancel, hold 2 s for SOS, hold 5 s for power. B: confirm medication / cancel.",
         size="3.5 × 2.9 × 1.35 mm side-actuated SMD switches + printed plungers",
         where="Board edge, +Y side, outside the strap; plungers through the side wall",
         links="GPIOs on the ESP32-S3", note="Print the plungers in TPU, or mould them into a soft overlay.",
         pieces=[piece for x in P["btn_x"] for piece in (
             (box(x - 1.75, x + 1.75, 11.6, 14.5, PCB1, PCB1 + 1.35), "#2b2d31", 0.0),
             (cyl_y(x, zb, 1.6, 14.5, 16.0), "#3c3f45", 0.0))]),
    dict(id="p-ppg", name="Optical sensor MAXM86161", short="MAXM86161", layer="bot", ex=4, dim="2.9 × 4.3",
         role="Heart rate, SpO₂ and breathing rate; a module designed for the wrist (LEDs + photodiode + front end).",
         size="2.9 × 4.3 × 1.4 mm", where="Skin side of the PCB, flush in the floor window",
         links="I²C + interrupt to the ESP32-S3",
         note="Add a clear window (epoxy or a small lens) so sweat and dirt stay out.",
         pieces=[(box(P["ppg_x"] - 1.45, P["ppg_x"] + 1.45, -2.15, 2.15, 0.0, PCB0), "#15151a", 0.0)]),
    dict(id="p-pogo", name="Pogo charging pads ×4", short="Pogo pads", layer="bot", ex=4, dim="Ø1.8",
         role="Magnetic clip-on charger: 5 V, GND and USB D+/D− (charging and flashing through one connector).",
         size="4 gold pads, 3 mm pitch", where="Skin side of the PCB, under 4 holes in the floor",
         links="Charger input and the ESP32-S3's native USB",
         note="Gold-plated (ENIG) pads; a 4-pin magnetic pogo cable costs a few dollars.",
         pieces=[(cyl_z(P["pogo_x"], y, 0.9, PCB0 - 0.05, PCB0, 24), GOLD, 0.8) for y in P["pogo_y"]]),
    dict(id="p-lipo", name="LiPo 402530 (3.7 V)", short="LiPo", layer="lid", ex=18, dim="30 × 25 × 4",
         role="Power: about 180–200 mAh. Needs the low-power firmware (deep sleep, WiFi only for alerts) for a full day.",
         size="30 × 25 × 4 mm pouch", where="On top of the board's parts, inside the lid, clear of the antenna end",
         links="Soldered to the battery pads", note="A 3 mm cell (e.g. 302530) makes the pod about 9 mm thick.",
         pieces=[(box(-18.8, 11.2, -12.5, 12.5, B0, B1), "#b4bec9", 0.25),
                 (box(-18.8, -15.8, -11.0, 11.0, B0 + 0.4, B1 - 0.4), "#d9a21b", 0.0)]),
]


def fit_check(base, lid):
    ok = True
    case = base + lid
    print(f"{'part':<28} {'vs base':>8} {'vs lid':>8} {'gap':>7}")
    solids = []
    for p in PARTS:
        s = p["pieces"][0][0]
        for piece in p["pieces"][1:]:
            s += piece[0]
        solids.append((p["name"], s))
        vb, vl = (s ^ base).volume(), (s ^ lid).volume()
        flag = "" if vb < EPS and vl < EPS else "  <-- COLLIDES"
        ok &= not flag
        print(f"{p['name']:<28} {vb:>8.2f} {vl:>8.2f} {s.min_gap(case, 5.0):>5.2f} mm{flag}")
    for i in range(len(solids)):
        for j in range(i + 1, len(solids)):
            v = (solids[i][1] ^ solids[j][1]).volume()
            if v >= EPS:
                ok = False
                print(f"  overlap {solids[i][0]} x {solids[j][0]}: {v:.2f} mm^3")
    return ok


def hex_rgba(h, a=1.0):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (a,)


def main():
    base, lid = build()
    for label, part in (("base", base), ("lid", lid)):
        assert len(part.decompose()) == 1, f"pcb {label} is not one piece"
    assert (base ^ lid).volume() < 1e-3, "base and lid overlap"
    ok = fit_check(base, lid)

    lid_print = lid.rotate([180, 0, 0])
    lid_print = lid_print.translate([0, 0, -lid_print.bounding_box()[2]])
    write_stl(base, OUT / "case_base_pcb.stl")
    write_stl(lid_print, OUT / "case_lid_pcb.stl")
    items = [("Case base", (0.56, 0.70, 0.84, 0.35), base), ("Case lid", (0.65, 0.76, 0.87, 0.30), lid)]
    for p in PARTS:
        for k, (s, col, _) in enumerate(p["pieces"]):
            items.append((f"{p['name']} {k + 1}" if len(p["pieces"]) > 1 else p["name"], hex_rgba(col), s))
    write_glb(items, OUT / "assembly_pcb.glb")

    # viewer: case meshes (keys pbase / plid) + part meshes with their info
    js = OUT / "viewer" / "case_meshes.js"
    entries = dict(re.findall(r'(\w+): "([A-Za-z0-9+/=]*)"', js.read_text())) if js.exists() else {}
    entries["pbase"] = base64.b64encode(stl_bytes(base)).decode()
    entries["plid"] = base64.b64encode(stl_bytes(lid)).decode()
    js.write_text("// generated by make_case.py / make_case_xiao.py / make_case_pcb.py - binary STL (assembled frame, mm), base64\n"
                  "window.CASE_MESHES = {" + ", ".join(f'{k}: "{v}"' for k, v in entries.items()) + "};\n")
    out = []
    for p in PARTS:
        meta = {k: v for k, v in p.items() if k != "pieces"}
        meta["swatch"] = p["pieces"][0][1] if p["id"] != "p-pcb" else "#1f5f3a"
        meta["pieces"] = [{"stl": base64.b64encode(stl_bytes(s)).decode(), "color": col, "metal": m}
                          for s, col, m in p["pieces"]]
        out.append(meta)
    (OUT / "viewer" / "pcb_parts.js").write_text(
        "// generated by make_case_pcb.py - concept custom-PCB parts (binary STL base64 + info)\n"
        "window.PCB_PARTS = " + json.dumps(out, ensure_ascii=False) + ";\n")

    bb = base.bounding_box()
    L, W = P["inner_l"] + 2 * P["wall"], P["inner_w"] + 2 * P["wall"]
    print(f"pcb concept pod {L:.1f} x {W:.1f} mm (across wrist incl. strap bars {bb[4] - bb[1]:.1f} mm), "
          f"height {CEIL + P['lid_top']:.1f} mm; genus {base.genus()}/{lid.genus()}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
