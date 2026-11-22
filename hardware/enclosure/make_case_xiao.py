#!/usr/bin/env python3
"""
make_case_xiao.py - Compact wrist pod for the Seeed XIAO ESP32-S3 build of the Vital wristband.

    pip install manifold3d numpy matplotlib
    python hardware/enclosure/make_case_xiao.py      # writes case_*_xiao.stl + preview, updates viewer meshes

Compared with the DevKit pod (make_case.py, 64 x 46 x 27.4 mm) this one drops the ESP32 DevKit,
the DS3231 and the TP4056 (the XIAO charges the LiPo itself, the time comes from NTP), and
stacks the parts in three thin layers:

  lid,  top layer  (5.5 mm) : XIAO ESP32-S3 (USB-C through the -X end), FPC antenna on the lid,
                              coin vibration motor, 9 x 4.2 mm buzzer under the sound holes,
                              2 tactile buttons pressed through the top (glued under small bosses)
  base, battery    (6 mm)   : LiPo 602535 (35 x 25 x 6) on foam tape
  base, skin layer (4 mm)   : MAX30102 face-down over the floor window | MPU6050 | driver strip
                              (2 NPN, resistors, diode, battery divider) | slide power switch
                              with its lever through the -Y wall
  skin

X runs along the forearm (+X toward the hand), the strap bars sit on the long sides (+-Y) for a
22 mm strap. Every dimension is a parameter below: measure your parts, edit, re-run.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
from manifold3d import Manifold

from make_case import OUT, box, cyl_x, preview, rrect, stl_bytes, write_stl

# ----------------------------------------------------------------------------- parameters (mm)
P = dict(
    # cavity (inside): battery 35 x 25 and the skin layer (MAX30102 20 x 15 beside MPU6050 20.3 x 15.6)
    # inner corner radius 1 mm: a 3 mm radius clips the corners of the MPU6050, driver strip and switch
    inner_l=44.0, inner_w=30.0, inner_r=1.0,
    wall=2.0, floor=1.8, lid_top=1.6, chamfer=1.6,
    # vertical stack: skin layer + foam + battery live in the base, the XIAO layer in the lid
    skin_h=4.0, foam_h=0.5, battery_h=6.0, top_h=5.5,
    # lid locating rim
    rim_h=2.5, rim_t=1.2, fit=0.2,
    # MAX30102 window in the floor
    win_l=10.0, win_w=8.0, win_x=-10.5, win_y=0.0,
    # slide switch (SS12D00) lever slot in the -Y wall, between the corner and the strap cheek (|x| >= 14.5)
    sw_x=-16.7, sw_l=4.4, sw_h=2.8, sw_z=3.6,
    # XIAO USB-C: the lid's -X wall is thinned to `usb_wall` around the port so a plug seats fully
    usb_y=-4.7, usb_w=9.6, usb_h=3.6, usb_z=15.6, usb_wall=1.4, usb_pocket_w=20.0,
    # top-press buttons (6 x 6 tactile, 7-8 mm tall) glued under bosses on the lid ceiling
    btn_xy=((6.5, 8.5), (14.5, 8.5)), btn_d=3.8, boss=7.0, boss_h=1.2,
    # buzzer 9 x 4.2 mm, sound holes in the lid top
    buzz_x=16.1, buzz_y=-3.5, buzz_hole=1.4,
    # strap bars for a 22 mm strap on both long sides
    strap_w=22.0, bar_out=6.0, bar_gap=3.2, bar_d=3.0, bar_cheek=3.0, bar_h=6.0,
)


def build(p: dict):
    L, W = p["inner_l"] + 2 * p["wall"], p["inner_w"] + 2 * p["wall"]
    R = p["inner_r"] + p["wall"]
    hb = p["floor"] + p["skin_h"] + p["foam_h"] + p["battery_h"]        # base height
    c = p["chamfer"]

    # ---- base: chamfered skin-side edges
    shell = Manifold.batch_hull([rrect(L - 2 * c, W - 2 * c, R - c).extrude(0.01),
                                 rrect(L, W, R).extrude(c)]) + rrect(L, W, R).extrude(hb - c).translate([0, 0, c])
    base = shell - rrect(p["inner_l"], p["inner_w"], p["inner_r"]).extrude(hb).translate([0, 0, p["floor"]])
    base -= rrect(p["win_l"], p["win_w"], 1.5).extrude(p["floor"] + 2).translate([p["win_x"], p["win_y"], -1])
    base -= box(p["sw_x"] - p["sw_l"] / 2, p["sw_x"] + p["sw_l"] / 2, -W / 2 - 1, -W / 2 + p["wall"] + 1,
                p["sw_z"] - p["sw_h"] / 2, p["sw_z"] + p["sw_h"] / 2)
    sw, out, gap = p["strap_w"] + 1.0, p["bar_out"], p["bar_gap"]
    for s in (1, -1):
        y0, y1 = sorted((s * (W / 2 - 0.5), s * (W / 2 + out)))
        for xs in (-1, 1):
            base += box(*sorted((xs * sw / 2, xs * (sw / 2 + p["bar_cheek"]))), y0, y1, 0, p["bar_h"])
        base += cyl_x(s * (W / 2 + gap + p["bar_d"] / 2), p["bar_h"] / 2, p["bar_d"], -sw / 2 - 0.5, sw / 2 + 0.5)

    # ---- lid (assembly frame, sits on the base at z = hb)
    ht, ceil = p["top_h"], hb + p["top_h"]
    ri_l, ri_w = p["inner_l"] - 2 * p["fit"], p["inner_w"] - 2 * p["fit"]
    li_l, li_w = ri_l - 2 * p["rim_t"], ri_w - 2 * p["rim_t"]
    r_in = max(0.5, p["inner_r"] - p["fit"] - p["rim_t"])
    lid = rrect(L, W, R).extrude(ht + p["lid_top"]).translate([0, 0, hb]) - \
        rrect(li_l, li_w, r_in).extrude(ht + 0.01).translate([0, 0, hb - 0.01])
    rim = rrect(ri_l, ri_w, p["inner_r"] - p["fit"]).extrude(p["rim_h"] + 0.01) - \
        rrect(li_l, li_w, r_in).extrude(p["rim_h"] + 1).translate([0, 0, -0.5])
    lid += rim.translate([0, 0, hb - p["rim_h"]])
    # thin the -X wall around the USB-C port, then cut the port (rounded slot)
    x_pocket = -L / 2 + p["usb_wall"]
    lid -= box(x_pocket, -li_l / 2 + 0.5, p["usb_y"] - p["usb_pocket_w"] / 2, p["usb_y"] + p["usb_pocket_w"] / 2,
               hb - p["rim_h"] - 1, ceil - 0.01)
    port = rrect(p["usb_h"], p["usb_w"], p["usb_h"] / 2 - 0.01).extrude(p["usb_wall"] + 2).rotate([0, 90, 0])
    lid -= port.translate([-L / 2 - 1, p["usb_y"], p["usb_z"]])
    # button bosses on the ceiling, then the button holes through boss + top
    for bx, by in p["btn_xy"]:
        lid += box(bx - p["boss"] / 2, bx + p["boss"] / 2, by - p["boss"] / 2, by + p["boss"] / 2, ceil - p["boss_h"], ceil + 0.01)
        lid -= Manifold.cylinder(p["boss_h"] + p["lid_top"] + 2, p["btn_d"] / 2, circular_segments=40).translate(
            [bx, by, ceil - p["boss_h"] - 1])
    # buzzer sound holes (hex pattern of 7)
    pts = [(0, 0)] + [(2.6 * np.cos(a), 2.6 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    for dx, dy in pts:
        lid -= Manifold.cylinder(p["lid_top"] + 2, p["buzz_hole"] / 2, circular_segments=24).translate(
            [p["buzz_x"] + dx, p["buzz_y"] + dy, ceil - 1])
    return base, lid, hb, ceil + p["lid_top"]


def main():
    base, lid, hb, htot = build(P)
    for label, part in (("base", base), ("lid", lid)):
        assert part.status().name == "NoError", part.status()
        assert len(part.decompose()) == 1, f"xiao {label} is not one piece - check the geometry"
    assert (base ^ lid).volume() < 1e-3, "base and lid overlap"
    lid_print = lid.rotate([180, 0, 0])
    lid_print = lid_print.translate([0, 0, -lid_print.bounding_box()[2]])
    write_stl(base, OUT / "case_base_xiao.stl")
    write_stl(lid_print, OUT / "case_lid_xiao.stl")
    preview(base, lid, OUT / "preview_xiao.png", titles=(
        "Exploded view (XIAO compact pod)", "Base from the skin side: sensor window, switch slot, strap bars",
        "Base, inside: skin layer + battery bay", "Lid, inside: button bosses, buzzer holes, USB-C (-X)"))

    # add the assembled meshes to the 3D viewer's mesh file (keys xbase / xlid)
    js = OUT / "viewer" / "case_meshes.js"
    entries = dict(re.findall(r'(\w+): "([A-Za-z0-9+/=]*)"', js.read_text())) if js.exists() else {}
    entries["xbase"] = base64.b64encode(stl_bytes(base)).decode()
    entries["xlid"] = base64.b64encode(stl_bytes(lid)).decode()
    js.write_text("// generated by make_case.py / make_case_xiao.py - binary STL (assembled frame, mm), base64\n"
                  "window.CASE_MESHES = {" + ", ".join(f'{k}: "{v}"' for k, v in entries.items()) + "};\n")

    bb = base.bounding_box()
    print(f"xiao pod {bb[3] - bb[0]:.1f} x {P['inner_w'] + 2 * P['wall']:.1f} mm "
          f"(across wrist incl. strap bars {bb[4] - bb[1]:.1f} mm), height {htot:.1f} mm (base {hb:.1f}); "
          f"base {base.volume() / 1000:.1f} cm3, lid {lid.volume() / 1000:.1f} cm3, genus {base.genus()}/{lid.genus()}")


if __name__ == "__main__":
    main()
