#!/usr/bin/env python3
"""
build_space.py - Package the 3D assembly viewer as a Hugging Face Static Space.

    python hardware/enclosure/viewer/build_space.py

Writes web_3d/ at the repository root: index.html (the viewer wrapped in a full HTML document so it
renders correctly on phones), the two mesh data files and the Space README. Upload the four files to a
Static Space (see web_3d/README.md, or the steps printed below). Re-run after changing a case script.
"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "web_3d"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Interactive 3D model of the Vital wristband pod: every component placed inside the case.">
<style>html,body{margin:0}[hidden]{display:none!important}</style>
</head>
<body>
"""

README = """---
title: Vital Wristband 3D Pod
emoji: ⌚
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
short_description: 3D model of the wristband pod with every part inside
pinned: false
---

# Vital Wristband: 3D pod assembly

Interactive 3D model of the Vital wristband pod with every component placed where it goes inside the
3D-printed case. Switch between three versions:

* **XIAO** (default): compact pod on a Seeed XIAO ESP32-S3, 48 × 34 × 19.4 mm
* **Custom PCB**: future-work concept, one custom board, 43 × 33 × 10.3 mm
* **DevKit**: original pod on an ESP32 DevKit, 64 × 46 × 27.4 mm

Drag to rotate, scroll to zoom, use the slider to explode the layers, and click a part for its size,
position and wiring. *Wrist + strap* shows the pod on a typical wrist.

Generated from the parametric case scripts in
https://github.com/f20230104-dotcom/Vital-armband/tree/main/hardware/enclosure
"""


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(HEAD + (HERE / "index.html").read_text() + "\n</body>\n</html>\n")
    for f in ("case_meshes.js", "pcb_parts.js"):
        shutil.copy2(HERE / f, OUT / f)
    (OUT / "README.md").write_text(README)
    print(f"Space files written to {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<16} {f.stat().st_size / 1024:7.1f} kB")


if __name__ == "__main__":
    main()
