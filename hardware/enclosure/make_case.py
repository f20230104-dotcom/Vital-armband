#!/usr/bin/env python3
"""
make_case.py - Parametric 3D-printable wrist pod for the SmartVitals / Vital wristband.

    pip install manifold3d numpy matplotlib
    python hardware/enclosure/make_case.py            # writes STLs + preview PNG next to this file

Two parts per variant, both exported in print orientation (no supports needed):
  case_base_<variant>.stl  open-top tub: sensor window, battery rails, ESP32 USB opening, strap bars
  case_lid_<variant>.stl   cap with locating rim: TP4056 USB-C opening, 2 button holes, buzzer holes

Variants
  standard : LiPo inside the pod (stack: battery 6 mm + ESP32 8 mm + top modules 9 mm)
  slim     : LiPo in a pocket under the strap, pod ~7 mm thinner

Sized for a typical adult wrist (circumference 16-18 cm, top-of-wrist width ~55-60 mm):
the pod's long axis (X) runs ALONG THE FOREARM and the strap bars sit on the long sides, so the
lug-to-lug width across the wrist is only ~56 mm. Strap: any 22 mm Velcro / nylon sport loop.

Every dimension is a parameter below - measure your parts and edit, then re-run.

Layout (top view, X along the forearm, +X = ESP32 USB end):
  base, bottom layer : MAX30102 over the floor window at x=-19 | LiPo 35x25 at x=-6..29 in rails
  base, middle layer : ESP32 DevKit (pins removed) on foam tape over both, USB at the +X wall
  lid,  top layer    : +Y half: DS3231 (x=-28..10), buzzer (x=16), motor (x=24), buttons on the +Y wall
                       -Y half: TP4056 (x=-28..-2, USB-C at the -X wall), MPU6050 (x=-1..20)
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

OUT = Path(__file__).resolve().parent
SEG = 48                      # circle resolution

# ----------------------------------------------------------------------------- parameters (mm)
P = dict(
    # cavity (inside) - fits ESP32 DevKit 55 x 28 with headers removed
    inner_l=60.0, inner_w=42.0, inner_r=3.0,
    wall=2.0, floor=1.8, lid_top=1.6,
    # vertical stack inside the base (standard variant)
    battery_h=6.0, gap_h=1.0, esp32_h=8.0,
    # top layer inside the lid (DS3231 without pins ~8-9 mm, MPU6050, TP4056, buzzer)
    top_h=9.0,
    # lid locating rim (goes down into the base)
    rim_h=2.5, rim_t=1.2, fit=0.2,
    # comfort: chamfer on the skin-side edges
    chamfer=1.8,
    # MAX30102 window in the floor (sensor chip ~5.6 x 3.3 mm on a ~14-20 mm board)
    win_l=10.0, win_w=8.0, win_x=-19.0,
    # battery 602535 (35 x 25 x 6) rails, battery placed toward +X
    batt_l=35.0, batt_w=25.0, batt_x=11.5, rail_h=2.0, rail_t=1.4,
    # ESP32 USB opening (micro-USB or USB-C) in the +X end wall
    usb_w=12.0, usb_h=7.0,
    # TP4056 USB-C opening in the -X end of the lid skirt
    tp_w=10.0, tp_h=5.0, tp_y=-11.1,
    # side buttons (6x6 mm tactile, 7-8 mm tall) on the +Y side of the lid
    btn_d=3.6, btn_x=(14.0, 24.0),
    # buzzer sound holes in the lid top
    buzz_x=16.0, buzz_y=4.0, buzz_hole=1.6,
    # strap bars for a 22 mm strap on both long sides
    strap_w=22.0, bar_out=6.0, bar_gap=3.2, bar_d=3.0, bar_cheek=3.0, bar_h=6.0,
)
VARIANTS = {"standard": {}, "slim": {"battery_h": 0.0, "gap_h": 0.0}}


# ----------------------------------------------------------------------------- helpers
def rrect(l, w, r):
    """Rounded rectangle centred on the origin (CrossSection)."""
    r = max(0.01, min(r, l / 2 - 0.01, w / 2 - 0.01))
    return CrossSection.square([l - 2 * r, w - 2 * r], center=True).offset(r, JoinType.Round, circular_segments=SEG)


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_y(x, z, d, y0, y1):
    """Cylinder along Y from y0 to y1 at (x, z)."""
    return Manifold.cylinder(y1 - y0, d / 2, circular_segments=SEG).rotate([-90, 0, 0]).translate([x, y0, z])


def cyl_x(y, z, d, x0, x1):
    return Manifold.cylinder(x1 - x0, d / 2, circular_segments=SEG).rotate([0, 90, 0]).translate([x0, y, z])


def write_stl(m: Manifold, path: Path):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    t = np.asarray(mesh.tri_verts).astype(np.int64)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"SmartVitals wrist pod".ljust(80, b" "))
        f.write(struct.pack("<I", len(t)))
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3), ("c", "<f4", 3), ("x", "<u2")])
        rec["n"], rec["a"], rec["b"], rec["c"] = n, a, b, c
        f.write(rec.tobytes())


# ----------------------------------------------------------------------------- parts
def build(p: dict):
    L, W = p["inner_l"] + 2 * p["wall"], p["inner_w"] + 2 * p["wall"]
    R = p["inner_r"] + p["wall"]
    depth = p["battery_h"] + p["gap_h"] + p["esp32_h"]
    hb = p["floor"] + depth                                  # base height
    c = p["chamfer"]

    # ---- base shell: chamfered skin-side edges via hull of two slabs
    shell = Manifold.batch_hull([rrect(L - 2 * c, W - 2 * c, R - c).extrude(0.01),
                                 rrect(L, W, R).extrude(c).translate([0, 0, 0])]) + \
        rrect(L, W, R).extrude(hb - c).translate([0, 0, c])
    cavity = rrect(p["inner_l"], p["inner_w"], p["inner_r"]).extrude(hb).translate([0, 0, p["floor"]])
    base = shell - cavity

    # sensor window through the floor
    base -= rrect(p["win_l"], p["win_w"], 1.5).extrude(p["floor"] + 2).translate([p["win_x"], 0, -1])

    # battery rails + end stop (standard variant only)
    if p["battery_h"] > 0:
        bx0, bx1 = p["batt_x"] - p["batt_l"] / 2, p["batt_x"] + p["batt_l"] / 2
        yy = p["batt_w"] / 2 + p["fit"] * 2
        z0, z1 = p["floor"] - 0.01, p["floor"] + p["rail_h"]
        for s in (1, -1):
            y_in, y_out = s * yy, s * (yy + p["rail_t"])
            base += box(bx0 + 4, bx1 - 2, min(y_in, y_out), max(y_in, y_out), z0, z1)
        base += box(bx1 + p["fit"], bx1 + p["fit"] + p["rail_t"], -yy, yy, z0, z1)

    # ESP32 USB opening in the +X wall, centred on the board's height
    board_z = p["floor"] + p["battery_h"] + p["gap_h"]
    base -= box(L / 2 - p["wall"] - 1, L / 2 + 1, -p["usb_w"] / 2, p["usb_w"] / 2,
                board_z - 0.5, min(board_z - 0.5 + p["usb_h"], hb - 1.5))

    # strap bars on both long sides: two cheeks + a round bar the strap loops around
    sw, out, gap = p["strap_w"] + 1.0, p["bar_out"], p["bar_gap"]
    for s in (1, -1):
        y_wall = s * (W / 2 - 0.5)
        y_end = s * (W / 2 + out)
        y0, y1 = min(y_wall, y_end), max(y_wall, y_end)
        for xs in (-1, 1):
            x_in = xs * sw / 2
            x_out = xs * (sw / 2 + p["bar_cheek"])
            base += box(min(x_in, x_out), max(x_in, x_out), y0, y1, 0, p["bar_h"])
        bar_y = s * (W / 2 + gap + p["bar_d"] / 2)
        base += cyl_x(bar_y, p["bar_h"] / 2, p["bar_d"], -sw / 2 - 0.5, sw / 2 + 0.5)

    # ---- lid (assembly frame: sits on top of the base at z = hb)
    ht = p["top_h"]
    # the lid's inner wall is thicker than the base wall by (fit + rim_t), so its inside face is
    # flush with the rim's inside face and the rim is fused to the skirt (one printable piece)
    ri_l, ri_w = p["inner_l"] - 2 * p["fit"], p["inner_w"] - 2 * p["fit"]
    li_l, li_w = ri_l - 2 * p["rim_t"], ri_w - 2 * p["rim_t"]
    lid = rrect(L, W, R).extrude(ht + p["lid_top"]).translate([0, 0, hb]) - \
        rrect(li_l, li_w, max(0.5, p["inner_r"] - p["fit"] - p["rim_t"])).extrude(ht).translate([0, 0, hb - 0.01])
    rim = rrect(ri_l, ri_w, p["inner_r"] - p["fit"]).extrude(p["rim_h"] + 0.01) - \
        rrect(ri_l - 2 * p["rim_t"], ri_w - 2 * p["rim_t"], max(0.5, p["inner_r"] - p["fit"] - p["rim_t"])).extrude(p["rim_h"] + 1).translate([0, 0, -0.5])
    lid += rim.translate([0, 0, hb - p["rim_h"]])
    # notch the rim where the ESP32 USB opening is, so the plug clears it
    lid -= box(L / 2 - p["wall"] - p["rim_t"] - 1, L / 2, -p["usb_w"] / 2 - 1, p["usb_w"] / 2 + 1, hb - p["rim_h"] - 1, hb + 0.01)
    # TP4056 USB-C opening in the -X end of the skirt
    lid -= box(-L / 2 - 1, -li_l / 2 + 1, p["tp_y"] - p["tp_w"] / 2, p["tp_y"] + p["tp_w"] / 2,
               hb + 1.0, hb + 1.0 + p["tp_h"])
    # side buttons on the +Y skirt
    for bx in p["btn_x"]:
        lid -= cyl_y(bx, hb + ht / 2, p["btn_d"], li_w / 2 - 1, W / 2 + 1)
    # buzzer holes (hex pattern of 7)
    zt = hb + ht - 1
    pts = [(0, 0)] + [(3.2 * np.cos(a), 3.2 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    for dx, dy in pts:
        lid -= Manifold.cylinder(p["lid_top"] + 2, p["buzz_hole"] / 2, circular_segments=24).translate(
            [p["buzz_x"] + dx, p["buzz_y"] + dy, zt])
    return base, lid, hb, hb + ht + p["lid_top"]


def mesh_arrays(m):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)


def preview(base, lid, path):
    """Shaded multi-view render (matplotlib) for checking the design without a CAD program."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def draw(ax, m, rgb, el, az, light=(0.3, -0.5, 0.8)):
        v, t = mesh_arrays(m)
        tri = v[t]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        e, a = np.radians(el), np.radians(az)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        keep = n @ cam > 0                                  # back-face culling
        tri, n = tri[keep], n[keep]
        lv = np.asarray(light) / np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip(np.abs(n @ lv), 0, 1)
        cols = np.clip(np.outer(shade, rgb), 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors="none", zsort="max"))

    blue, orange = np.array([0.46, 0.76, 0.90]), np.array([0.99, 0.69, 0.09])
    bb = base.bounding_box()
    base_flipped = base.rotate([180, 0, 0])
    fb = base_flipped.bounding_box()
    base_flipped = base_flipped.translate([0, 0, -fb[2]])
    lid_inside = lid.rotate([180, 0, 0])
    lb = lid_inside.bounding_box()
    lid_inside = lid_inside.translate([0, 0, -lb[2]])
    views = [
        ("Exploded view", [(base, blue), (lid.translate([0, 0, 16]), orange)], (24, -58), (-6, 52)),
        ("Base from the skin side: sensor window, chamfer, strap bars", [(base_flipped, blue)], (58, -60), (-6, 30)),
        ("Base, inside: battery rails, USB opening (+X)", [(base, blue)], (58, -128), (-6, 30)),
        ("Lid, inside: rim, button holes (+Y), USB-C (-X)", [(lid_inside, orange)], (58, -60), (-6, 30)),
    ]
    fig = plt.figure(figsize=(16, 4.6))
    for k, (title, parts, (el, az), zl) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        for m, rgb in parts:
            draw(ax, m, rgb, el, az)
        ax.set_xlim(-38, 38); ax.set_ylim(-38, 38); ax.set_zlim(*zl)
        ax.set_box_aspect((1, 1, (zl[1] - zl[0]) / 76))
        ax.view_init(elev=el, azim=az)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    report = []
    for name, over in VARIANTS.items():
        p = {**P, **over}
        base, lid, hb, htot = build(p)
        for label, part in (("base", base), ("lid", lid)):
            assert part.status().name == "NoError", part.status()
            pieces = len(part.decompose())
            assert pieces == 1, f"{name} {label} is {pieces} separate pieces - check the geometry"
        # lid printed upside down: top plate on the bed
        lid_print = lid.rotate([180, 0, 0])
        bb = lid_print.bounding_box()
        lid_print = lid_print.translate([0, 0, -bb[2]])
        write_stl(base, OUT / f"case_base_{name}.stl")
        write_stl(lid_print, OUT / f"case_lid_{name}.stl")
        bbb = base.bounding_box()
        report.append(f"{name:<8} pod {bbb[3] - bbb[0]:.1f} x {p['inner_w'] + 2 * p['wall']:.1f} mm "
                      f"(across wrist incl. strap bars {bbb[4] - bbb[1]:.1f} mm), height {htot:.1f} mm; "
                      f"base {base.volume() / 1000:.1f} cm3, lid {lid.volume() / 1000:.1f} cm3, genus {base.genus()}/{lid.genus()}")
        if name == "standard":
            preview(base, lid, OUT / "preview_standard.png")
    print("\n".join(report))


if __name__ == "__main__":
    main()
