#!/usr/bin/env python3
"""
build_presentation.py - Builds docs/Vital_Wristband_BITS_F312.pptx (short deck: 1-2 slides per section) in the BITS Pilani Dubai
Campus course template (4:3, 720 x 540 pt): clock-tower title slide with the navy band and crest,
content slides with the "innovate / achieve / lead" tabs and the tri-colour footer.

All numbers are read from the experiment outputs, so re-running after re-training refreshes them:
    ml_training/models/metrics.json, history.json      (deployed model)
    ml_training/results/experiments.json, trigger.json (evaluation study)

    python docs/presentation/build_presentation.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = HERE / "assets"
SHOTS = ROOT / "docs" / "screenshots"
OUT = ROOT / "docs" / "Vital_Wristband_BITS_F312.pptx"

NAVY, ORANGE, SKY, RED = RGBColor(0x10, 0x11, 0x41), RGBColor(0xFC, 0xB0, 0x17), RGBColor(0x76, 0xC2, 0xE5), RGBColor(0xE3, 0x1C, 0x24)
INK, MUTED, DARKRED = RGBColor(0x24, 0x26, 0x3A), RGBColor(0x5A, 0x5E, 0x78), RGBColor(0xB4, 0x14, 0x1B)
WHITE, TINT, TINT2, LINE = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF3, 0xF4, 0xF9), RGBColor(0xEE, 0xF6, 0xFC), RGBColor(0xD9, 0xDC, 0xE8)
HILITE, PALE = RGBColor(0xFF, 0xF4, 0xD6), RGBColor(0xCF, 0xE3, 0xF5)
FONT = "Arial"

M = json.loads((ROOT / "ml_training/models/metrics.json").read_text())
H = json.loads((ROOT / "ml_training/models/history.json").read_text())
E = json.loads((ROOT / "ml_training/results/experiments.json").read_text())
T = json.loads((ROOT / "ml_training/results/trigger.json").read_text())
FW = json.loads((ROOT / "ml_training/results/firmware_sim.json").read_text())

prs = Presentation()
prs.slide_width, prs.slide_height = Pt(720), Pt(540)
BLANK = prs.slide_layouts[6]
_n = [0]


# ============================================================================ primitives
def _runs(p, content, size, color, bold):
    if isinstance(content, str):
        content = [(content, {})]
    for txt, o in content:
        r = p.add_run()
        r.text = txt
        f = r.font
        f.name = FONT
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.italic = o.get("italic", False)
        f.color.rgb = o.get("color", color)


def _bullet(p, level=0):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Pt(14 + 14 * level))))
    pPr.set("indent", str(int(-Pt(12))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•")


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, after=0, bullets=False, line=None, name=None):
    tb = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str) or (isinstance(paras, list) and paras and isinstance(paras[0], tuple)):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        if line:
            p.line_spacing = line
        _runs(p, para, size, color, bold)
        if bullets:
            _bullet(p)
    return tb


def box(slide, x, y, w, h, fill, line_color=None, rounded=False, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Pt(x), Pt(y), Pt(w), Pt(h))
    if rounded:
        shp.adjustments[0] = 0.08
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(1.25)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def card(slide, x, y, w, h, title, body, fill=TINT, title_color=NAVY, size=12, line_color=None):
    box(slide, x, y, w, h, fill, line_color, rounded=True)
    text(slide, x + 12, y + 10, w - 24, 18, title, size=size + 2, bold=True, color=title_color)
    text(slide, x + 12, y + 32, w - 24, h - 40, body, size=size, color=INK, line=1.1)


def arrow(slide, x, y, w, h=14, color=NAVY):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Pt(x), Pt(y), Pt(w), Pt(h))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    a.shadow.inherit = False
    return a


def table(slide, x, y, w, widths, rows, size=11, header_fill=NAVY, highlight_last=False, row_h=20,
          align=None, bold_first_col=False):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Pt(x), Pt(y), Pt(w), Pt(row_h * nr))
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    tot = sum(widths)
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Pt(w * cw / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Pt(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Pt(5)
            c.margin_top = c.margin_bottom = Pt(3)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
            elif highlight_last and i == nr - 1:
                c.fill.fore_color.rgb = HILITE
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else TINT
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[j] if align else PP_ALIGN.LEFT)
            bold = i == 0 or (highlight_last and i == nr - 1) or (bold_first_col and j == 0)
            _runs(p, str(val), size, WHITE if i == 0 else INK, bold)
    return gf


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def style_chart(chart, legend=True, size=10):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = INK
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


def axis_style(chart, vmin=None, vmax=None, fmt="0.00", major=None):
    va = chart.value_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    if major:
        va.major_unit = major
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.color.rgb = MUTED
    ca = chart.category_axis
    ca.tick_labels.font.color.rgb = INK
    ca.format.line.color.rgb = LINE


def skip_labels(chart, n: int) -> None:
    """Show every n-th category label / tick (crowded category axes)."""
    cat = chart._chartSpace.find(".//" + qn("c:catAx"))
    anchor = cat.find(qn("c:noMultiLvlLbl"))
    for tag in ("c:tickLblSkip", "c:tickMarkSkip"):
        el = etree.Element(qn(tag))
        el.set("val", str(n))
        if anchor is not None:
            anchor.addprevious(el)
        else:
            cat.append(el)


def bar_chart(slide, x, y, w, h, cats, series: dict, colors, vmin=0, vmax=1, fmt="0.00", labels=True,
              kind=XL_CHART_TYPE.COLUMN_CLUSTERED, size=10, legend=True, gap=60, label_fmt=None, major=None):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, [float(v) for v in vals])
    gf = slide.shapes.add_chart(kind, Pt(x), Pt(y), Pt(w), Pt(h), cd)
    ch = gf.chart
    style_chart(ch, legend=legend and len(series) > 1, size=size)
    axis_style(ch, vmin, vmax, fmt, major)
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.overlap = 0 if kind != XL_CHART_TYPE.BAR_CLUSTERED else 0
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = label_fmt or fmt
        dl.number_format_is_linked = False
        dl.font.size = Pt(size - 1)
        dl.font.color.rgb = INK
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    return ch


def line_chart(slide, x, y, w, h, cats, series: dict, colors, vmin=None, vmax=None, fmt="0.00", size=10,
               legend=True, widths=None):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, [float(v) for v in vals])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE, Pt(x), Pt(y), Pt(w), Pt(h), cd)
    ch = gf.chart
    style_chart(ch, legend=legend and len(series) > 1, size=size)
    axis_style(ch, vmin, vmax, fmt)
    for k, (s, c) in enumerate(zip(ch.plots[0].series, colors)):
        s.smooth = False
        s.marker.style = XL_MARKER_STYLE.NONE
        s.format.line.color.rgb = c
        s.format.line.width = Pt((widths or [2.25] * 9)[k])
    return ch


# ============================================================================ template frames
def content_slide(eyebrow: str, title: str):
    """White content slide in the template's style."""
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    s.shapes.add_picture(str(ASSETS / "header_tabs.png"), Pt(516), Pt(0), Pt(204), Pt(64.5))
    box(s, 0, 74, 26, 4, ORANGE, name="Title marker")
    for x0, x1, c in ((164, 360, ORANGE), (360, 540, SKY), (540, 720, RED)):
        box(s, x0, 515, x1 - x0, 3, c, name="Footer line")
    text(s, 450, 523, 252, 13, [("BITS", {"bold": True}), (" Pilani, Dubai Campus", {})], size=10.5, color=NAVY,
         align=PP_ALIGN.RIGHT, name="Footer")
    text(s, 36, 523, 120, 13, str(_n[0]), size=10, color=MUTED, name="Slide number")
    text(s, 36, 40, 470, 14, eyebrow.upper(), size=10, bold=True, color=DARKRED, name="Eyebrow")
    text(s, 36, 60, 648, 34, title, size=23, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    return s


def screenshot_slide(eyebrow, title, img, caption_title, caption, note):
    s = content_slide(eyebrow, title)
    box(s, 91, 105, 538, 338, TINT, LINE)
    s.shapes.add_picture(str(SHOTS / img), Pt(92), Pt(106), Pt(536), Pt(335))
    text(s, 92, 450, 536, 16, caption_title, size=13, bold=True, color=NAVY)
    text(s, 92, 468, 536, 32, caption, size=11.5, color=MUTED, line=1.05)
    notes(s, note)
    return s


def pct(v, d=1):
    return f"{100 * v:.{d}f} %"


# ============================================================================ slides
# ============================================================================ slides (short deck)
kS, kU = "SisFall + synthetic test (12 subj.)", "UCI real-wrist test (5 subj.)"


def title_slide(heading, lines, note, heading_size=24, lines_y=350):
    """Template title slide: clock tower, navy band, crest."""
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    s.shapes.add_picture(str(ASSETS / "title_bg.png"), 0, 0, Pt(720), Pt(540))
    box(s, 0, 264, 684, 216, NAVY, name="Title band")
    for x0, x1, c in ((0, 228, ORANGE), (228, 456, SKY), (456, 684, RED)):
        box(s, x0, 480, x1 - x0, 6, c)
    s.shapes.add_picture(str(ASSETS / "crest.png"), Pt(22), Pt(276), Pt(96), Pt(92))
    text(s, 12, 372, 150, 32, [("BITS", {"bold": True}), (" Pilani", {})], size=27, color=WHITE)
    text(s, 14, 404, 140, 14, "Dubai Campus", size=11, color=WHITE)
    text(s, 160, 276, 510, 64, heading, size=heading_size, bold=True, color=WHITE, line=1.0)
    y = lines_y
    for txt, size, bold, col in lines:
        text(s, 160, y, 510, 18, txt, size=size, bold=bold, color=col)
        y += size + 10
    notes(s, note)
    return s


# 1 ---------------------------------------------------------------- cover
title_slide("Vital Wristband: TinyML Fall Detection and Vital-Sign Monitoring on ESP32",
            [("Course No: BITS F312 · Neural Networks & Fuzzy Logic", 14, True, WHITE),
             ("Course Instructor: Dr. V. Kalaichelvi, Associate Professor/EEE", 14, True, WHITE),
             ("Presented by: [Student name (ID)], [Student name (ID)], [Student name (ID)]", 12, False, PALE)],
            "A low-cost ESP32 wristband that detects falls on the device, monitors heart rate, SpO2 and breathing, "
            "reminds the wearer about medication and alerts a caregiver. Software complete; hardware due 10 December.")

# 2 ---------------------------------------------------------------- 1. introduction
s = content_slide("1 · Introduction", "Why a fall-detecting wristband?")
for k, (num, lab, c) in enumerate((("684,000", "fall deaths worldwide every year", RED),
                                   ("37.3 million", "falls a year that need medical care", ORANGE))):
    y0 = 124 + k * 120
    box(s, 36, y0, 5, 88, c)
    text(s, 54, y0 + 4, 290, 48, num, size=40, bold=True, color=NAVY)
    text(s, 54, y0 + 56, 290, 24, lab, size=14, color=MUTED)
text(s, 36, 372, 300, 14, "Source: WHO “Falls” fact sheet, 2021", size=9.5, color=MUTED)
for k, (t_, d_) in enumerate((("Fall detection", "on-device TinyML"), ("Vital signs", "HR · SpO₂ · breathing"),
                              ("Medication", "reminders + escalation"), ("Caregiver", "live alert dashboard"))):
    x0, y0 = 372 + (k % 2) * 160, 124 + (k // 2) * 120
    box(s, x0, y0, 152, 108, TINT2, rounded=True)
    text(s, x0 + 14, y0 + 30, 124, 20, t_, size=15, bold=True, color=NAVY)
    text(s, x0 + 14, y0 + 56, 124, 36, d_, size=12, color=INK)
notes(s, "WHO: about 684,000 people die from falls each year and 37.3 million falls need medical attention. Lying "
         "unnoticed after a fall is what makes it dangerous. Our band detects falls, watches vitals, handles "
         "medication reminders and alerts a caregiver.")

# 3 ---------------------------------------------------------------- 2. problem statement
s = content_slide("2 · Problem statement", "Problem statement")
box(s, 36, 110, 648, 70, TINT)
box(s, 36, 110, 5, 70, NAVY)
text(s, 56, 110, 612, 70, "Detect falls and abnormal vitals on a ≈ US$30 wristband, without the cloud, "
                          "and alert a caregiver within seconds.", size=17, bold=True, color=NAVY,
     anchor=MSO_ANCHOR.MIDDLE, line=1.1)
for k, (t_, d_) in enumerate((("Too expensive", "Fall-detecting watches cost US$300–400, plus fees"),
                              ("False alarms", "Simple thresholds fire on jumps and slaps"),
                              ("Incomplete", "No missed-dose alerts; breathing tracked only in sleep"))):
    card(s, 36 + k * 220, 200, 208, 100, t_, d_, fill=WHITE, line_color=LINE, size=12.5)
text(s, 36, 322, 648, 16, "Constraints", size=13, bold=True, color=NAVY)
for k, t_ in enumerate(("520 KB RAM", "20 ms per sample", "Battery powered", "Works offline")):
    x0 = 36 + k * 164
    box(s, x0, 346, 156, 40, HILITE, rounded=True)
    text(s, x0, 346, 156, 40, t_, size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "We want fall and vital-sign alerts on a cheap wristband, computed on the device, reaching a caregiver "
         "in seconds. Existing options are expensive, false-alarm too often, or miss features like medication "
         "escalation.")

# 4 ---------------------------------------------------------------- 3. literature: market
s = content_slide("3 · Literature review", "Wristbands on the market vs ours")
rows = [["Product", "Price", "Fall detection", "SpO₂", "Breathing", "Missed-dose alert", "Monthly fee"],
        ["Apple Watch Series 10", "US$399", "Yes", "Yes", "Sleep only", "No", "No"],
        ["Samsung Galaxy Watch7", "US$299", "Yes", "Yes", "Sleep only", "No", "No"],
        ["Google Pixel Watch 3", "US$349", "Yes", "Yes", "Sleep only", "No", "Optional"],
        ["Fitbit Charge 6", "US$160", "No", "Yes", "Sleep only", "No", "Optional"],
        ["Medical-alert watch", "US$100–200", "Add-on", "Varies", "No", "No", "US$30–45"],
        ["Budget band (Noise, boAt)", "US$20–50", "No", "Spot check", "No", "No", "No"],
        ["Vital Wristband (ours)", "≈ US$30", "On-device", "Yes", "Continuous", "Yes", "No"]]
table(s, 36, 108, 648, [24, 13, 14, 11, 12, 15, 11], rows, size=11, highlight_last=True, row_h=30,
      align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6)
text(s, 36, 362, 648, 20, [("Ours: ", {"bold": True}), ("about 1/10 the cost, no subscription, all four functions.", {})],
     size=14, color=NAVY)
text(s, 36, 390, 648, 14, "Approximate US launch prices; verify before citing.", size=9.5, color=MUTED)
notes(s, "Premium watches detect falls but cost 300 to 400 dollars and track breathing only during sleep. None "
         "alert a caregiver about a missed dose. Budget bands lack fall detection; alert services charge monthly.")

# 5 ---------------------------------------------------------------- 3. literature: research
s = content_slide("3 · Literature review", "Research: what was missing")
rows = [["Work", "Approach", "Gap"],
        ["Bourke et al., 2007", "Fixed acceleration thresholds", "Many false alarms"],
        ["Özdemir & Barshan, 2014", "6 body sensors, ML on a PC", "Not wearable, no vitals"],
        ["Sucerquia et al., 2017 (SisFall)", "Waist sensor, thresholds", "Not on the wrist"],
        ["Musci et al., 2021", "LSTM network", "Too heavy for a microcontroller"],
        ["This project", "Wrist IMU + PPG, 14 KB CNN on ESP32", "Tested on unseen people"]]
table(s, 36, 108, 648, [34, 38, 28], rows, size=12, highlight_last=True, row_h=34)
text(s, 36, 330, 648, 40, [("Gap we fill: ", {"bold": True}),
                           ("one wrist device, small model on the chip, tested on people it never saw.", {})],
     size=14, color=NAVY)
notes(s, "Earlier work was threshold-based, ran on a PC, used waist sensors, or used networks too large for a "
         "microcontroller. We combine a wrist device, an on-chip model and testing on unseen people.")

# 6 ---------------------------------------------------------------- 4. design: system + hardware
s = content_slide("4 · Design and implementation", "System design and hardware")
for k, (t_, d_, f_, l_) in enumerate((("Sensors", "IMU · PPG · RTC\nbuttons · battery", TINT2, SKY),
                                      ("ESP32 (FreeRTOS)", "fall model · vitals DSP\nreminders · WiFi alerts", TINT, NAVY),
                                      ("Caregiver", "webhook server\nlive dashboard", HILITE, ORANGE))):
    x0 = 36 + k * 228
    card(s, x0, 106, 192, 84, t_, d_, fill=f_, line_color=l_, size=12)
    if k < 2:
        arrow(s, x0 + 198, 140, 26)
rows = [["Component", "Purpose", "≈ US$"],
        ["ESP32-WROOM-32 board", "Processor + WiFi", "6"],
        ["MPU6050", "Motion (fall detection)", "2"],
        ["MAX30102", "Heart rate, SpO₂, breathing", "3"],
        ["DS3231 RTC", "Medication alarms", "3"],
        ["Buttons, buzzer, vibration motor", "Cancel, SOS, alerts", "2.5"],
        ["LiPo 500 mAh + charger", "Power", "6"],
        ["Case, strap, board", "Wearable", "7"],
        ["Total", "", "≈ 30"]]
table(s, 36, 206, 648, [44, 42, 14], rows, size=11, highlight_last=True, row_h=22,
      align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
notes(s, "Sensors feed the ESP32, which runs four FreeRTOS tasks: fall detection with the neural network, vital "
         "signs, medication reminders and WiFi alerts to a caregiver dashboard. All parts cost about 30 dollars.")

# 7 ---------------------------------------------------------------- 4. design: dashboard scenarios
s = content_slide("4 · Design and implementation", "Dashboard: four scenarios")
shots = [("02_dashboard_fall_alert.jpg", "Fall confirmed → caregiver alert"),
         ("04_demo_fall_countdown.jpg", "Fall detected → 15 s cancel countdown"),
         ("05_demo_jump_rejected.jpg", "Jump → rejected, no alert"),
         ("06_demo_hypoxia_alert.jpg", "SpO₂ 87 % for 30 s → vitals alert")]
for k, (img, cap) in enumerate(shots):
    x0, y0 = 68 + (k % 2) * 304, 102 + (k // 2) * 204
    box(s, x0 - 1, y0 - 1, 282, 177, TINT, LINE)
    s.shapes.add_picture(str(SHOTS / img), Pt(x0), Pt(y0), Pt(280), Pt(175))
    text(s, x0, y0 + 179, 280, 16, cap, size=11.5, bold=True, color=NAVY)
notes(s, "Four scenarios from the dashboard and live demo: a confirmed fall alert, the 15-second countdown the "
         "wearer can cancel, a jump correctly ignored, and a low-oxygen alert after 30 seconds.")

# 8 ---------------------------------------------------------------- 5. experiments: data + model
s = content_slide("5 · Experiments", "Data and model")
rows = [["Dataset", "People", "Windows", "Use"],
        ["SisFall (waist)", "38", f"{M['dataset_sources']['sisf']:,}", "Train / test"],
        ["UCI falls (right wrist)", "17", f"{M['dataset_sources']['uci']:,}", "12 train, 5 unseen"],
        ["Synthetic wrist", "—", f"{M['dataset_sources']['syn']:,}", "Augmentation"]]
table(s, 36, 106, 400, [38, 12, 16, 34], rows, size=11.5, row_h=28)
for k, (num, lab) in enumerate(((f"{M['params']:,}", "parameters"), (f"{M['tflite_bytes'] / 1024:.1f} KB", "int8 model"),
                                (f"{M['macs'] / 1000:.0f} k", "operations / window"))):
    y0 = 106 + k * 38
    text(s, 456, y0, 110, 30, num, size=20, bold=True, color=NAVY)
    text(s, 570, y0 + 6, 114, 20, lab, size=12, color=MUTED)
text(s, 36, 240, 648, 16, "Detection pipeline", size=13, bold=True, color=NAVY)
steps = [("Trigger", "impact > 2 g", NAVY, WHITE), ("Stillness", "2.5 s", NAVY, WHITE),
         ("1D-CNN", "p(fall) ≥ 0.5", RED, WHITE), ("Countdown", "15 s to cancel", ORANGE, NAVY),
         ("Alert", "to dashboard", ORANGE, NAVY)]
for k, (t_, d_, f_, c_) in enumerate(steps):
    x0 = 36 + k * 132
    box(s, x0, 262, 116, 62, f_, rounded=True)
    text(s, x0, 270, 116, 20, t_, size=13, bold=True, color=c_, align=PP_ALIGN.CENTER)
    text(s, x0, 294, 116, 20, d_, size=11, color=c_, align=PP_ALIGN.CENTER)
    if k < 4:
        arrow(s, x0 + 118, 287, 12, 12)
text(s, 36, 344, 648, 70, ["Windows of 2.56 s at 50 Hz; split by person, never by window",
                           "Training adds random rotations, so sensor mounting doesn’t matter",
                           "Vitals: band-pass filters (HR 0.5–4 Hz, breathing 0.1–0.5 Hz) + motion gate"],
     size=12, bullets=True, after=4)
notes(s, "Three datasets: SisFall, real wrist recordings from the UCI dataset, and synthetic wrist data. The model "
         "is a small 1D CNN, 13.8 KB after int8 quantization. It only runs after a cheap trigger and a stillness "
         "check, then the wearer gets 15 seconds to cancel.")

# 9 ---------------------------------------------------------------- 5. experiments: what we tested
s = content_slide("5 · Experiments", "What we tested")
ab = E["ablation"]
bar_chart(s, 36, 104, 410, 300, ["SisFall only", "+ synthetic", "+ rotation", "+ wrist data", "+ both (final)"],
          {"Recall on unseen wrist data": [ab[n][kU]["recall"] for n in ab]}, [ORANGE], vmin=0.6, vmax=1.02,
          fmt="0.0", label_fmt="0.00", size=10, gap=50, major=0.1, legend=False)
text(s, 36, 404, 410, 16, "Ablation: recall on 5 unseen wrist volunteers", size=11, color=MUTED, align=PP_ALIGN.CENTER)
base = E["baselines"]
rows = [["Method", "F1 (wrist)", "Size"],
        ["Threshold rules", f"{base['Threshold rules'][kU]['f1']:.2f}", "tiny"],
        ["Random Forest", f"{base['Random Forest (42 feat.)'][kU]['f1']:.2f}", "1.4 MB"],
        ["CNN int8 (ours)", f"{base['1D-CNN int8 (deployed)'][kU]['f1']:.2f}", "14 KB"]]
table(s, 462, 110, 222, [48, 26, 26], rows, size=11, row_h=28, highlight_last=True,
      align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
cv = E["cv"]
box(s, 462, 250, 222, 80, TINT, rounded=True)
text(s, 462, 258, 222, 34, f"{100 * cv['accuracy_mean']:.1f} ± {100 * cv['accuracy_std']:.1f} %", size=22, bold=True,
     color=NAVY, align=PP_ALIGN.CENTER)
text(s, 462, 296, 222, 20, "5-fold cross-validation accuracy", size=11, color=MUTED, align=PP_ALIGN.CENTER)
notes(s, "Ablation: random-rotation augmentation and real wrist data lift recall on unseen wrist volunteers from 80 "
         "to 98 percent. Baselines: thresholds alone fail; a Random Forest is decent but 1.4 MB; our CNN is best and "
         "14 KB. Five-fold cross-validation by person gives about 99 percent accuracy.")

# 10 --------------------------------------------------------------- 6. results
s = content_slide("6 · Results", "Results on 17 people the model never saw")
ia = M["int8"]["all"]
for k, (num, lab) in enumerate(((pct(ia["accuracy"]), "accuracy"), (pct(ia["precision"]), "precision"),
                                (pct(ia["recall"]), "recall"), (f"{ia['auc']:.3f}", "ROC AUC"))):
    x0 = 36 + k * 164
    box(s, x0, 108, 156, 92, TINT, rounded=True)
    text(s, x0, 120, 156, 40, num, size=26, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    text(s, x0, 164, 156, 20, lab, size=12, color=MUTED, align=PP_ALIGN.CENTER)
rows = [["Test set", "Accuracy", "Recall"]]
for key, lab in (("sisf", "SisFall (12 people)"), ("uci", "Real wrist (5 people)")):
    rows.append([lab, pct(M["int8"][key]["accuracy"]), pct(M["int8"][key]["recall"])])
table(s, 36, 220, 400, [52, 24, 24], rows, size=12, row_h=30, align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
tr0 = T["free fall <0.5 g (60 ms) then impact >2.7 g (spec)"]["UCI real-wrist test"]["trigger_recall_falls"]
tr1 = T["spec path OR impact >2.0 g (deployed)"]["UCI real-wrist test"]["trigger_recall_falls"]
box(s, 456, 220, 228, 90, HILITE, rounded=True)
text(s, 470, 228, 200, 18, "Key fix found in testing", size=12, bold=True, color=NAVY)
text(s, 470, 250, 200, 56, f"Falls reaching the model: {100 * tr0:.0f}% → {100 * tr1:.0f}% with an extra impact trigger",
     size=12, color=INK, line=1.1)
for k, (num, lab) in enumerate(((f"{FW['falls_detected']}/{FW['falls_total']}", "simulated falls caught"),
                                (f"{FW['false_triggers']}/{FW['hard_negatives']}", "false alerts (jumps, slaps)"),
                                (f"{FW['hr_mae']:.1f} bpm", "heart-rate error"))):
    x0 = 36 + k * 220
    text(s, x0, 336, 208, 30, num, size=20, bold=True, color=NAVY)
    text(s, x0, 366, 208, 18, lab, size=11.5, color=MUTED)
notes(s, "On 9,052 windows from 17 unseen people the int8 model reaches about 98 percent accuracy, precision and "
         "recall, equally good on real wrist data. Testing also exposed a design flaw: the original trigger let "
         "only 44 percent of real falls reach the model; an extra impact trigger raised that to 99 percent. In an "
         "80-minute firmware simulation every fall was caught with no false alerts.")

# 11 --------------------------------------------------------------- 6. results: hardware feasibility
s = content_slide("6 · Results", "Will it run on the hardware?")
rows = [["Resource", "Budget", "Needs", "Status"],
        ["Flash", "1.9 MB", "1.08 MB", "Fits"],
        ["RAM", "320 KB", "86 KB", "Fits"],
        ["Inference", "< 20 ms", "≈ 5–15 ms (est.)", "Measure on device"],
        ["Battery", "≥ 24 h", "≈ 9–12 h (est.)", "Needs power tuning"]]
table(s, 36, 108, 648, [22, 22, 30, 26], rows, size=12.5, row_h=34, bold_first_col=True,
      align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT])
for k, (t_, d_) in enumerate((("Ready", "Firmware builds for ESP32; 15/15 tests pass; watchdog and sensor checks added"),
                              ("Needs hardware", "Real timing, wrist PPG quality, battery life, own wrist data"))):
    card(s, 36 + k * 332, 296, 316, 96, t_, d_, fill=TINT if k == 0 else HILITE, size=12.5)
notes(s, "Memory fits easily: about half the flash and a quarter of RAM. Inference time and battery life are "
         "estimates; the firmware logs real values at boot. Battery life is the main open risk.")

# 12 --------------------------------------------------------------- 7. future work: plan
s = content_slide("7 · Future work", "Plan to a working wristband by 10 December")
rows = [["Week", "Tasks", "Done when"],
        ["12–18 Nov", "Buy parts, breadboard, flash firmware", "All sensors detected"],
        ["19–25 Nov", "Record wrist data, calibrate sensors", "300+ labelled trials"],
        ["26 Nov–2 Dec", "Fine-tune model, build case and strap", "Wearable prototype"],
        ["3–9 Dec", "Fall, wear and battery tests", "Recall ≥ 95 %"],
        ["10 Dec", "Final demonstration", "Working wristband"]]
table(s, 36, 108, 648, [20, 46, 34], rows, size=12.5, row_h=36, highlight_last=True, bold_first_col=True)
text(s, 36, 340, 648, 16, "Test falls only by healthy volunteers onto crash mats.", size=10.5, color=MUTED)
notes(s, "Four weeks: parts and breadboard, wrist data and calibration, fine-tuning and the wearable build, then "
         "formal tests, ending with the demonstration on 10 December.")

# 13 --------------------------------------------------------------- 7. future work: risks
s = content_slide("7 · Future work", "Risks and next steps")
rows = [["Risk", "Plan"], ["Short battery life", "Lower clock, sleep between samples"],
        ["Noisy wrist PPG", "Tune LED, motion gate"], ["Board too big", "Smaller ESP32-S3 module"],
        ["Little real fall data", "Record volunteers on mats"]]
table(s, 36, 108, 400, [46, 54], rows, size=12, row_h=34)
text(s, 456, 108, 228, 18, "After December", size=14, bold=True, color=NAVY)
text(s, 456, 134, 228, 160, ["Phone app over BLE", "Alerts outdoors (LTE-M)", "Over-the-air updates",
                             "Trials with older adults"], size=13, bullets=True, after=8)
notes(s, "Battery life is the main risk. After the course: a phone app, outdoor connectivity, OTA updates and "
         "approved trials with older adults.")

# 14 --------------------------------------------------------------- thank you
title_slide("Thank you", [("Questions are welcome.", 15, False, PALE),
                          ("Live demo: huggingface.co/spaces/kkshik/vital-wristband", 13, False, WHITE),
                          ("Code: github.com/f20230104-dotcom/Vital-armband", 13, False, WHITE)],
            "Thank you. The live demo and all code are online.", heading_size=36, lines_y=330)

# speaker notes: three-person script (docs/presentation/speaker_script.py)
import sys as _sys
_sys.path.insert(0, str(HERE))
from speaker_script import SCRIPT  # noqa: E402
assert len(SCRIPT) == len(prs.slides), "one script entry per slide"
for _slide, (_who, _secs, _words) in zip(prs.slides, SCRIPT):
    notes(_slide, f"SPEAKER {_who} · about {_secs} s\n\n{_words}")

prs.save(OUT)
print(f"wrote {OUT} ({len(prs.slides)} slides)")
