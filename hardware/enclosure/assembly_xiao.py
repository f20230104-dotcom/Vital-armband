#!/usr/bin/env python3
"""
assembly_xiao.py - Every component of the compact (XIAO ESP32-S3) pod as a solid, placed in the case.

    python hardware/enclosure/assembly_xiao.py

1. Fit check: each part against the base, the lid and every other part (interference volume), plus the
   smallest gap between each part and the case. Fails (exit 1) if anything collides.
2. Exports assembly_xiao.glb: case + all parts in colour, assembled, in metres (glTF Y-up). Opens in
   Windows 3D Viewer, Blender, Fusion 360 (as mesh), or any online glTF viewer.

Positions match make_case_xiao.py and the 3D viewer (viewer/index.html, "Compact" version). Part sizes are
typical seller dimensions: measure yours and edit PARTS.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from manifold3d import Manifold

from make_case import OUT, box
from make_case_xiao import P, build, variant


def cyl_z(cx, cy, r, z0, z1, seg=40):
    return Manifold.cylinder(z1 - z0, r, circular_segments=seg).translate([cx, cy, z0])


def u(*ms):
    out = ms[0]
    for m in ms[1:]:
        out += m
    return out


# heights (mm) - see make_case_xiao.py
FL, SKIN, B_LO, B_HI, CEIL = 1.8, 5.8, 6.3, 12.1, 17.8


def make_parts(p: dict):
    """(name, RGBA, solid) for every part of the given make_case_xiao variant (one solid per physical part)."""
    boss = CEIL - p["boss_h"]                                # where the button bodies are glued
    mx, my = p["motor_xy"]
    bb, bh, act = p["btn_body"] / 2, p["btn_body_h"], p["btn_act"]
    stem = p["btn_total_h"] - bh

    def button(x, y):
        body = box(x - bb, x + bb, y - bb, y + bb, boss - bh, boss)
        if p["btn_sq"] > 0:                                  # square lit actuator (12 x 12 LED switch)
            top = box(x - act / 2, x + act / 2, y - act / 2, y + act / 2, boss, boss + stem)
            legs = [box(*sorted((x + sx * bb, x + sx * (bb + 1.2))), y + dy - 0.3, y + dy + 0.3, boss - bh, boss - bh + 0.3)
                    for sx in (-1, 1) for dy in (-3.75, 0.0, 3.75)]          # 4 switch + 2 LED legs, bent flat
        else:                                                # round stem (6 x 6 switch)
            top = cyl_z(x, y, act / 2, boss, boss + stem, 28)
            legs = [box(x + dx - 0.3, x + dx + 0.3, y + dy - 0.6, y + dy + 0.6, boss - bh, boss - bh + 0.3)
                    for dx, dy in ((-2.2, -3.6), (2.2, -3.6), (-2.2, 3.6), (2.2, 3.6))]   # legs bent flat, along Y
        return u(body, top, *legs)

    return [
        # ---- lid layer
        ("XIAO ESP32-S3", (0.17, 0.19, 0.22, 1), u(
            box(-22.0, -1.0, -13.6, 4.2, 12.8, 14.0),                       # PCB 21 x 17.8
            box(-14.8, -3.8, -10.5, 1.1, 14.0, 16.4),                       # RF shield
            box(-22.5, -15.15, -4.7 - 4.47, -4.7 + 4.47, 14.0, 17.2),       # USB-C receptacle
            box(-3.4, -1.6, 1.4, 3.2, 14.0, 15.2))),                        # U.FL
        ("FPC antenna", (0.79, 0.54, 0.17, 1), box(-19, -2, 6.2, 12.8, CEIL - 0.2, CEIL)),
        ("Vibration motor 1027", (0.78, 0.80, 0.83, 1), cyl_z(mx, my, 5, CEIL - 2.8, CEIL)),
        ("Buzzer 9042", (0.15, 0.16, 0.17, 1), cyl_z(p["buzz_x"], p["buzz_y"], 4.5, CEIL - 4.2, CEIL)),
    ] + [
        (f"Button {n}", (0.24, 0.25, 0.27, 1), button(x, y)) for n, (x, y) in zip("AB", p["btn_xy"])
    ] + [
        # ---- battery layer
        ("LiPo 602535", (0.71, 0.75, 0.79, 1), u(
            box(-14.5, 17.5, -12.5, 12.5, B_LO, B_HI),
            box(-17.5, -14.5, -11, 11, B_LO + 0.4, B_HI - 0.6))),           # protection board end
        ("Foam under battery", (0.79, 0.76, 0.68, 1), box(-13, 15, -10, 10, SKIN, B_LO)),
        ("Foam under XIAO", (0.79, 0.76, 0.68, 1), box(-17, -2, -11.5, 2.5, B_HI, 12.8)),
        # ---- skin layer
        ("MAX30102", (0.36, 0.17, 0.53, 1), u(
            box(-20.5, -0.5, -7.5, 7.5, FL, FL + 1.6),
            box(-13.3, -7.7, -1.65, 1.65, 0.3, FL),                         # sensor chip in the floor window
            box(-6, -3.5, -5.5, 5.5, FL + 1.6, FL + 2.7))),
        ("MPU6050 GY-521", (0.16, 0.37, 0.68, 1), u(
            box(0.7, 21.0, -14.6, 1.0, FL, FL + 1.6),
            box(8.8, 12.8, -8.8, -4.8, FL + 1.6, FL + 2.5),
            box(15, 17, -3.5, -2, FL + 1.6, FL + 2.7))),
        ("Driver strip", (0.77, 0.58, 0.29, 1), u(
            box(0.7, 20.7, 2.0, 14.6, FL, FL + 1.6),
            *[box(x - 2.3, x + 2.3, 3.3, 7.5, FL + 1.6, FL + 3.9) for x in (4.2, 15.2)],          # BC547 laid flat
            *[box(x - 2.6, x + 2.6, y - 0.9, y + 0.9, FL + 1.6, FL + 3.4)                         # R / D
              for x, y in ((9.7, 5.4), (4.5, 11.2), (9.7, 11.2), (14.4, 11.2), (18.1, 11.2))])),
        ("Power switch SS12D00", (0.56, 0.58, 0.62, 1), u(
            box(p["sw_x"] - 4.3, p["sw_x"] + 4.3, -15, -11.4, FL, FL + 3.5),
            box(p["sw_x"] - 0.75, p["sw_x"] + 0.75, -17.25, -15, 2.8, 4.4),                       # lever in the slot
            *[box(x - 0.3, x + 0.3, -11.4, -9.0, FL + 1.4, FL + 2.1) for x in (p["sw_x"] - 2.5, p["sw_x"], p["sw_x"] + 2.5)])),
    ]


PARTS = make_parts(P)

EPS = 0.02          # mm^3: below this an "overlap" is just two faces touching


def fit_check(base, lid, parts=PARTS):
    ok = True
    print(f"{'part':<22} {'vs base':>9} {'vs lid':>9} {'gap to case':>12}")
    case = base + lid
    for name, _, solid in parts:
        vb, vl = (solid ^ base).volume(), (solid ^ lid).volume()
        gap = solid.min_gap(case, 5.0)
        flag = "" if vb < EPS and vl < EPS else "  <-- COLLIDES"
        ok &= not flag
        print(f"{name:<22} {vb:>8.2f}  {vl:>8.2f}  {gap:>10.2f} mm{flag}")
    clashes = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            v = (parts[i][2] ^ parts[j][2]).volume()
            if v >= EPS:
                clashes.append(f"  {parts[i][0]} x {parts[j][0]}: {v:.2f} mm^3")
    if clashes:
        ok = False
        print("part-to-part overlaps:\n" + "\n".join(clashes))
    else:
        print("part-to-part: no overlaps")
    return ok


def write_glb(items, path: Path):
    """Minimal binary glTF 2.0 writer: one mesh per item, flat-shaded (viewers compute normals)."""
    bin_chunks, buffer_views, accessors, meshes, nodes, materials = [], [], [], [], [], []
    offset = 0

    def add_view(data: bytes, target):
        nonlocal offset
        pad = (-len(data)) % 4
        bin_chunks.append(data + b"\0" * pad)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        offset += len(data) + pad
        return len(buffer_views) - 1

    for name, rgba, solid in items:
        mesh = solid.to_mesh()
        v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
        t = np.asarray(mesh.tri_verts).astype(np.uint32)
        # un-index so every face gets its own flat normal
        v = v[t.reshape(-1)]
        idx = np.arange(len(v), dtype=np.uint32)
        bv_p = add_view(v.tobytes(), 34962)
        bv_i = add_view(idx.tobytes(), 34963)
        accessors.append({"bufferView": bv_p, "componentType": 5126, "count": len(v), "type": "VEC3",
                          "min": v.min(0).tolist(), "max": v.max(0).tolist()})
        accessors.append({"bufferView": bv_i, "componentType": 5125, "count": len(idx), "type": "SCALAR"})
        mat = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0.0,
                                                      "roughnessFactor": 0.6}, "doubleSided": True}
        if rgba[3] < 1:
            mat["alphaMode"] = "BLEND"
        materials.append(mat)
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": len(accessors) - 2},
                                                     "indices": len(accessors) - 1, "material": len(materials) - 1}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    # root: mm -> m and Z-up -> glTF Y-up (rotate -90 deg about X)
    root = {"name": "Vital wristband compact pod", "children": list(range(len(nodes))),
            "scale": [0.001] * 3, "rotation": [-0.70710678, 0, 0, 0.70710678]}
    nodes.append(root)
    gltf = {"asset": {"version": "2.0", "generator": "assembly_xiao.py"}, "scene": 0,
            "scenes": [{"nodes": [len(nodes) - 1]}], "nodes": nodes, "meshes": meshes, "materials": materials,
            "accessors": accessors, "bufferViews": buffer_views, "buffers": [{"byteLength": offset}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binary = b"".join(bin_chunks)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def main():
    ok = True
    for name, suffix in (("xiao", "xiao"), ("xiao_led", "xiao_led")):
        p = variant(name)
        parts = make_parts(p)
        base, lid, hb, htot = build(p)
        print(f"== {name} ({p['inner_l'] + 2 * p['wall']:.0f} x {p['inner_w'] + 2 * p['wall']:.0f} x {htot:.1f} mm)")
        ok &= fit_check(base, lid, parts)
        items = [("Case base", (0.56, 0.70, 0.84, 0.35), base), ("Case lid", (0.65, 0.76, 0.87, 0.30), lid)] + parts
        write_glb(items, OUT / f"assembly_{suffix}.glb")
        print(f"wrote assembly_{suffix}.glb ({len(items)} bodies)\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
