#!/usr/bin/env python3
"""
build_presentation.py - Builds docs/Vital_Wristband_BITS_F312.pptx in the BITS Pilani Dubai
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
# 1 ---------------------------------------------------------------- cover (template title slide)
_n[0] += 1
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(str(ASSETS / "title_bg.png"), 0, 0, Pt(720), Pt(540))
box(s, 0, 264, 684, 216, NAVY, name="Title band")
for x0, x1, c in ((0, 228, ORANGE), (228, 456, SKY), (456, 684, RED)):
    box(s, x0, 480, x1 - x0, 6, c)
s.shapes.add_picture(str(ASSETS / "crest.png"), Pt(22), Pt(276), Pt(96), Pt(92))
text(s, 12, 372, 150, 32, [("BITS", {"bold": True}), (" Pilani", {})], size=27, color=WHITE)
text(s, 14, 404, 140, 14, "Dubai Campus", size=11, color=WHITE)
text(s, 160, 276, 510, 64, "Vital Wristband: TinyML Fall Detection and Vital-Sign Monitoring on ESP32",
     size=22, bold=True, color=WHITE, line=1.0)
text(s, 160, 350, 510, 16, "Course No: BITS F312 · Neural Networks & Fuzzy Logic", size=14, bold=True, color=WHITE)
text(s, 160, 372, 510, 16, "Course Instructor: Dr. V. Kalaichelvi, Associate Professor/EEE", size=14, bold=True,
     color=WHITE)
text(s, 160, 404, 510, 54, ["Presented by: [Student name (ID)], [Student name (ID)], [Student name (ID)]",
                            "November 2026"], size=12, color=PALE, after=4)
notes(s, "Our project is the Vital Wristband: a low-cost ESP32 wristband that detects falls with an on-device "
         "neural network, monitors heart rate, blood oxygen and breathing rate, reminds the wearer to take "
         "medication and alerts a caregiver dashboard. The software is complete and verified; the hardware build "
         "is planned for completion by 10 December 2026.")

# 2 ---------------------------------------------------------------- outline
s = content_slide("Outline", "Outline")
items = [("Introduction", "Why falls and vitals need watching"), ("Problem statement", "The gap we close"),
         ("Literature review", "Market wristbands and research"),
         ("Design and implementation", "Hardware, firmware, ML, dashboard"),
         ("Experiments", "Baselines, ablation, cross-validation"),
         ("Results", "Software results, hardware feasibility"), ("Future work", "Hardware plan to 10 Dec 2026")]
for k, (t_, d_) in enumerate(items):
    col, row = k // 4, k % 4
    x0, y0 = 36 + col * 330, 118 + row * 92
    b = box(s, x0, y0, 44, 44, NAVY, rounded=True)
    text(s, x0, y0, 44, 44, str(k + 1), size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x0 + 58, y0 + 2, 250, 20, t_, size=16, bold=True, color=NAVY)
    text(s, x0 + 58, y0 + 24, 250, 18, d_, size=12, color=MUTED)
notes(s, "Seven parts, in the order of the brief. Sections 4 to 6 carry most of the work.")

# 3 ---------------------------------------------------------------- introduction
s = content_slide("1 · Introduction", "Falls are common; lying unnoticed is the danger")
for k, (num, lab, c) in enumerate((("684,000", "deaths from falls worldwide each year, the second leading cause "
                                               "of unintentional injury death", RED),
                                   ("37.3 million", "falls a year severe enough to need medical attention; "
                                                    "adults over 60 suffer most", ORANGE))):
    y0 = 120 + k * 128
    box(s, 36, y0, 5, 104, c)
    text(s, 52, y0, 290, 50, num, size=40, bold=True, color=NAVY)
    text(s, 52, y0 + 54, 290, 50, lab, size=13, color=MUTED, line=1.05)
text(s, 36, 382, 300, 14, "Source: World Health Organization, “Falls” fact sheet, 26 April 2021.", size=9.5,
     color=MUTED)
text(s, 372, 118, 312, 20, "Our wristband in one sentence", size=16, bold=True, color=NAVY)
text(s, 372, 144, 312, 96, [("An ESP32 band that ", {}), ("detects falls on the device", {"bold": True}),
                            (" with a 14 KB neural network, tracks ", {}), ("heart rate, SpO₂ and breathing rate", {"bold": True}),
                            (", rings for ", {}), ("medication", {"bold": True}), (" and alerts a ", {}),
                            ("caregiver dashboard", {"bold": True}), (" over WiFi.", {})], size=14, line=1.15)
for k, (t_, d_) in enumerate((("Fall detection", "state machine + TinyML"), ("Vital signs", "PPG signal processing"),
                              ("Medication", "RTC alarms, escalation"), ("Caregiver view", "live web dashboard"))):
    x0, y0 = 372 + (k % 2) * 160, 262 + (k // 2) * 78
    card(s, x0, y0, 152, 68, t_, d_, fill=TINT2, size=11)
notes(s, "According to the WHO about 684,000 people die from falls each year and 37.3 million falls need medical "
         "attention. For older people living alone, lying on the floor for hours without help makes outcomes "
         "far worse. Our wristband detects the fall, lets the wearer cancel a false alarm, and otherwise alerts "
         "a caregiver within seconds. It also watches heart rate, blood oxygen and breathing, and reminds the "
         "wearer to take medication.")

# 4 ---------------------------------------------------------------- problem statement
s = content_slide("2 · Problem statement", "Reliable fall alerts cost too much or need the cloud")
box(s, 36, 108, 648, 82, TINT)
box(s, 36, 108, 5, 82, NAVY)
text(s, 54, 116, 618, 68, [("Design a low-cost wristband that detects falls ", {}), ("on the device", {"bold": True}),
                           (" with few false alarms, monitors ", {}), ("HR, SpO₂ and respiratory rate", {"bold": True}),
                           (", enforces ", {}), ("medication adherence", {"bold": True}),
                           (" and alerts a ", {}), ("caregiver", {"bold": True}),
                           (", within the memory and battery budget of an ESP32.", {})], size=15, line=1.15,
     anchor=MSO_ANCHOR.MIDDLE)
for k, (t_, d_) in enumerate((("Cost and subscriptions", "Smartwatches with fall detection cost US$300–400; "
                                                         "medical-alert services add a monthly fee."),
                              ("False alarms", "Threshold-only detectors fire on jumps, sit-downs and wrist "
                                               "slaps, so people stop wearing them."),
                              ("Missing pieces", "Budget bands have no fall detection and no caregiver "
                                                 "escalation for missed doses."))):
    card(s, 36 + k * 220, 206, 208, 118, t_, d_, fill=WHITE, line_color=LINE, size=12)
text(s, 36, 342, 648, 16, "Engineering constraints", size=13, bold=True, color=NAVY)
for k, t_ in enumerate(("520 KB SRAM · 4 MB flash", "50 Hz IMU: 20 ms per sample", "LiPo battery, radio mostly off",
                        "Works offline, no cloud inference")):
    x0 = 36 + k * 164
    box(s, x0, 364, 156, 40, HILITE, rounded=True)
    text(s, x0 + 8, 364, 140, 40, t_, size=11.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
text(s, 36, 420, 648, 80, ["Objectives", ], size=13, bold=True, color=NAVY)
text(s, 36, 440, 648, 70, ["Detect falls with ≥ 95 % recall on people the model has never seen, with few false alarms",
                           "Estimate HR, SpO₂ and breathing rate with motion rejection and persistence checks",
                           "Run fully in simulation now, and on a ≈ US$30 wristband by 10 December 2026"],
     size=12, bullets=True, after=3)
notes(s, "Build a low-cost wristband that detects falls on the device with few false alarms, monitors heart rate, "
         "SpO2 and breathing rate, enforces medication adherence and alerts a caregiver, within an ESP32's memory "
         "and battery budget. Three gaps motivate it: cost, false alarms and missing features.")

# 5 ---------------------------------------------------------------- literature: products
s = content_slide("3 · Literature review: wristbands on the market", "How existing wristbands compare with ours")
rows = [["Product", "Approx. price", "Fall detection", "SpO₂", "Breathing rate", "Missed dose → caregiver", "Monthly fee"],
        ["Apple Watch Series 10", "from US$399", "Yes", "Yes¹", "Sleep only", "No", "No"],
        ["Samsung Galaxy Watch7", "from US$299", "Yes", "Yes", "Sleep only", "No", "No"],
        ["Google Pixel Watch 3", "from US$349", "Yes", "Yes", "Sleep only", "No", "Optional"],
        ["Fitbit Charge 6", "US$160", "No", "Yes", "Sleep only", "No", "Optional"],
        ["Medical-alert watch²", "US$100–200", "Paid add-on", "Varies", "No", "No", "US$30–45"],
        ["Budget band (Noise, boAt)", "US$20–50", "No", "Spot check", "No", "No", "No"],
        ["Vital Wristband (ours)", "≈ US$30 parts", "On-device CNN", "Yes", "Continuous", "Yes, after 5 min", "No"]]
table(s, 36, 106, 648, [22, 14, 14, 9, 12, 16, 11], rows, size=10.5, highlight_last=True, row_h=24,
      align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6)
for k, (t_, d_) in enumerate((("Cost", "about one tenth of a fall-detecting smartwatch, no subscription"),
                              ("Features", "only option with falls, breathing rate and missed-dose escalation"),
                              ("Open", "on-device inference, open-source firmware, self-hosted dashboard"))):
    card(s, 36 + k * 220, 314, 208, 82, t_, d_, fill=TINT2, size=11.5)
text(s, 36, 410, 648, 40, ["Prices: approximate US launch prices of 2023–24 models; check before citing. "
                           "¹ Availability varies by region. ² e.g. Medical Guardian, Bay Alarm Medical; monitoring "
                           "fees are a typical range."], size=9.5, color=MUTED)
notes(s, "Premium smartwatches detect falls but cost 300 to 400 dollars, measure breathing only during sleep and "
         "none escalate a missed dose to a caregiver. Fitness and budget bands have no fall detection. "
         "Medical-alert watches charge a monthly fee. Our parts cost about 30 dollars with no subscription.")

# 6 ---------------------------------------------------------------- literature: research
s = content_slide("3 · Literature review: research", "What earlier work did, and what it left open")
rows = [["Work", "Sensors / placement", "Method", "Gap for a wristband"],
        ["Bourke et al., Gait & Posture, 2007", "Accelerometers, trunk and thigh", "Fixed thresholds",
         "Fires on hard daily activities"],
        ["Özdemir & Barshan, Sensors, 2014", "6 Xsens units incl. wrist (UCI data)", "k-NN, SVM, ANN offline",
         "PC-side, six sensors, no vitals"],
        ["Sucerquia et al., Sensors, 2017", "Waist device, 38 subjects (SisFall)", "Threshold features",
         "Waist, not wrist; no learned model"],
        ["Musci et al., IEEE TETC, 2021", "Waist IMU (SisFall)", "LSTM, online", "Recurrent nets costly on MCUs"],
        ["Charlton et al., IEEE RBME, 2018", "PPG and ECG", "Review: breathing rate", "Basis of our 0.1–0.5 Hz filter"],
        ["David et al., MLSys, 2021", "Microcontrollers", "TensorFlow Lite Micro", "The runtime we deploy on"],
        ["This project", "1 wrist IMU + PPG on ESP32", "Trigger + int8 1D-CNN + DSP",
         "Tested on unseen wrist subjects"]]
table(s, 36, 106, 648, [27, 26, 21, 26], rows, size=10.5, highlight_last=True, row_h=27)
text(s, 36, 344, 648, 60, [("Most published detectors are ", {}), ("waist-worn", {"bold": True}),
                           (" or run ", {}), ("off-device", {"bold": True}),
                           (". We pair a cheap trigger with a CNN that fits in 14 KB and test it on people it never saw.",
                            {})], size=14, line=1.15)
notes(s, "Early detectors used fixed thresholds. The UCI dataset compared classifiers offline with six sensors. "
         "SisFall, our main dataset, is waist-worn. Recurrent networks are heavy for a microcontroller. Charlton's "
         "review underpins our breathing-rate filter, and TensorFlow Lite Micro is our runtime.")

# 7 ---------------------------------------------------------------- architecture
s = content_slide("4 · Design and implementation", "System architecture")
card(s, 36, 112, 170, 180, "Sensors", "MPU6050 IMU, 50 Hz\nMAX30102 PPG, 25 Hz\nDS3231 RTC + EEPROM\nButtons A and B\n"
                                      "Battery voltage (ADC)", fill=TINT2, line_color=SKY, size=11.5)
arrow(s, 212, 196, 40)
text(s, 212, 178, 40, 14, "I²C", size=9.5, color=MUTED, align=PP_ALIGN.CENTER)
card(s, 258, 112, 222, 180, "ESP32 · FreeRTOS", "motion_task: fall trigger + int8 CNN\nvitals_task: HR, SpO₂, RR\n"
                                               "reminder_task: RTC alarms\nalert_task: queues → WiFi\n"
                                               "Buzzer, vibration, watchdog", fill=TINT, line_color=NAVY, size=11.5)
arrow(s, 486, 196, 40)
text(s, 486, 166, 40, 28, "WiFi\nJSON", size=9.5, color=MUTED, align=PP_ALIGN.CENTER)
card(s, 532, 112, 152, 180, "Caregiver side", "server.py: FastAPI webhook + SQLite\napp.py: Gradio dashboard\n"
                                              "Alerts, trends, acknowledge", fill=HILITE, line_color=ORANGE, size=11.5)
for k, (t_, d_) in enumerate((("Inference on the device", "Raw motion data never leaves the band; only events "
                                                          "and vitals are sent."),
                              ("Radio off by default", "WiFi wakes at once for a fall or SOS, otherwise every "
                                                       "5 minutes."),
                              ("Simulation mode", "Every sensor has a realistic mock, so all software runs "
                                                  "without hardware."))):
    x0 = 36 + k * 220
    box(s, x0, 316, 208, 3, (SKY, NAVY, ORANGE)[k])
    text(s, x0, 328, 208, 18, t_, size=13, bold=True, color=NAVY)
    text(s, x0, 350, 208, 60, d_, size=12, color=INK, line=1.1)
notes(s, "Sensors share one I2C bus. The ESP32 runs four FreeRTOS tasks. A Python webhook receiver stores events "
         "in SQLite and a Gradio dashboard shows them. Inference happens on the device, the radio stays off "
         "except to send, and every sensor has a simulated version.")

# 8 ---------------------------------------------------------------- hardware BOM
s = content_slide("4 · Design and implementation: hardware", "Hardware required: about US$30 of parts")
rows = [["Component", "Role", "Interface / ESP32 pin", "≈ US$"],
        ["ESP32-WROOM-32 dev board", "Dual-core MCU, WiFi", "I²C SDA 21, SCL 22", "6"],
        ["MPU6050 (GY-521)", "Accel + gyro: fall detection", "I²C 0x69 · INT GPIO27", "2"],
        ["MAX30102", "Red/IR PPG: HR, SpO₂, RR", "I²C 0x57 · INT GPIO32", "3"],
        ["DS3231 + AT24C32", "Dose alarms, event log", "I²C 0x68 / 0x50 · INT GPIO33", "3"],
        ["2 tactile buttons", "Cancel / SOS, confirm dose", "GPIO4, GPIO13", "0.5"],
        ["Active buzzer + vibration motor, 2 NPN", "Local alarm", "GPIO25, GPIO26", "2"],
        ["3.7 V LiPo 500 mAh + TP4056 charger", "Battery, USB charging", "100k/100k divider → GPIO34", "6"],
        ["Perfboard, 3D-printed case, strap", "Wearable enclosure", "n/a", "7"],
        ["Total", "≈ AED 110", "", "≈ 30"]]
table(s, 36, 106, 648, [33, 27, 30, 10], rows, size=11, highlight_last=True, row_h=25,
      align=[PP_ALIGN.LEFT] * 3 + [PP_ALIGN.RIGHT])
text(s, 36, 368, 648, 40, [("Address clash fixed in design: ", {"bold": True}),
                           ("the DS3231 module’s EEPROM ships at 0x57, the same as the MAX30102, so we bridge its "
                            "A0–A2 pads to move it to 0x50. Prices are approximate online retail.", {})],
     size=12, line=1.1)
notes(s, "Everything needed to build the wristband for about 30 US dollars. One detail caught in design: the RTC "
         "module's EEPROM defaults to 0x57, which clashes with the oximeter, so we move it to 0x50.")

# 9 ---------------------------------------------------------------- firmware
s = content_slide("4 · Design and implementation: firmware", "Four FreeRTOS tasks on two cores")
rows = [["Task", "Core", "Prio", "Period", "Job"],
        ["motion", "1", "5", "20 ms", "Read IMU, fall trigger, TFLite Micro inference, PPG motion gate"],
        ["vitals", "1", "4", "40 ms", "Drain MAX30102 FIFO; HR / SpO₂ / RR every second; 30 s persistence"],
        ["reminder", "0", "3", "250 ms", "DS3231 alarm interrupt, nag pattern, MED_TAKEN or MED_MISSED"],
        ["alert", "0", "2", "event", "Queue set (HIGH first); WiFi on → JSON POST → WiFi off; retries"],
        ["loop", "1", "1", "20 ms", "Buttons, SOS long-press, battery monitor, stack and heap stats"]]
table(s, 36, 106, 648, [13, 7, 7, 10, 63], rows, size=11, row_h=26, bold_first_col=True,
      align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT])
for k, (t_, d_) in enumerate((("Queues", "HIGH and LOW alert queues in a queue set; vitals in a 1-slot mailbox"),
                              ("Event group", "Motion gate, button presses, fall-alarm and dose-pending flags"),
                              ("Safety", "I²C mutex, task watchdog reboot, model self-test at boot"),
                              ("Build modes", "Simulation, real hardware, and a wrist data-logger"))):
    card(s, 36 + k * 164, 282, 156, 104, t_, d_, size=11)
text(s, 36, 400, 648, 40, "C++17 on Arduino-ESP32 2.0.17 (ESP-IDF 4.4) under PlatformIO; libraries: TensorFlowLite_ESP32, "
                          "Adafruit MPU6050, SparkFun MAX3010x, RTClib, ArduinoJson 7.", size=11, color=MUTED)
notes(s, "Sensing runs on core 1 at the highest priorities, so a slow WiFi connection never delays fall "
         "detection. Tasks communicate through priority queues, a mailbox and an event group. For hardware "
         "readiness we added an I2C mutex, a task watchdog, a model self-test at boot and a data-logger build.")

# 10 --------------------------------------------------------------- fall pipeline
s = content_slide("4 · Design and implementation: fall detection", "A cheap trigger, then the CNN decides")
steps = [("Trigger", "free fall then > 2.7 g, OR impact > 2.0 g", NAVY, WHITE),
         ("Stillness", "low motion for 2.5 s after the impact", NAVY, WHITE),
         ("CNN", "int8 1D-CNN, p(fall) ≥ 0.5", RED, WHITE),
         ("Countdown", "15 s buzz: any button cancels", ORANGE, NAVY),
         ("Alert", "FALL event to the dashboard", ORANGE, NAVY)]
for k, (t_, d_, f_, c_) in enumerate(steps):
    x0 = 36 + k * 132
    box(s, x0, 112, 116, 84, f_, rounded=True)
    text(s, x0 + 8, 120, 100, 18, t_, size=13, bold=True, color=c_)
    text(s, x0 + 8, 142, 100, 50, d_, size=10.5, color=c_, line=1.05)
    if k < 4:
        arrow(s, x0 + 118, 147, 12, 12)
tr = T["spec path OR impact >2.0 g (deployed)"]
tr0 = T["free fall <0.5 g (60 ms) then impact >2.7 g (spec)"]
text(s, 36, 214, 648, 18, "Why two trigger paths? Measured on held-out falls:", size=13, bold=True, color=NAVY)
rows = [["Trigger rule", "Falls triggered: SisFall", "Falls triggered: real wrist"],
        ["Free fall < 0.5 g for 60 ms, then impact > 2.7 g (original spec)",
         pct(tr0["SisFall test"]["trigger_recall_falls"]), pct(tr0["UCI real-wrist test"]["trigger_recall_falls"])],
        ["Spec path OR impact > 2.0 g without free fall (deployed)",
         pct(tr["SisFall test"]["trigger_recall_falls"]), pct(tr["UCI real-wrist test"]["trigger_recall_falls"])]]
table(s, 36, 236, 648, [56, 22, 22], rows, size=11, row_h=26, highlight_last=True,
      align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
text(s, 36, 330, 648, 70, ["Real falls often lack a clean free-fall phase: people grab furniture, slide or faint.",
                           "The extra path costs nothing: jumps, slaps and jogging fail the stillness check or the CNN.",
                           "The window is 128 samples (2.56 s at 50 Hz) with the impact at sample 64, exactly as in training."],
     size=12, bullets=True, after=4)
notes(s, "Every 20 ms sample goes through a cheap trigger. Our original specification required a free-fall "
         "valley below 0.5 g followed by an impact above 2.7 g, but on real held-out falls that path fires for "
         "only about 43 percent of them, because people grab, slide or faint. We kept it and added a second "
         "path: any impact above 2.0 g. Together they fire for about 99 percent of falls. The stillness check "
         "and the CNN then reject the extra triggers from jumps and slaps.")

# 11 --------------------------------------------------------------- CNN
s = content_slide("4 · Design and implementation: the model", "A 1D-CNN small enough for a microcontroller")
rows = [["Layer", "Output", "Params"], ["Input: 128 samples × 6 axes (g, °/s ÷ 500)", "128 × 6", "0"],
        ["Conv 8 @ 7, ReLU, max-pool 2", "64 × 8", "344"], ["Conv 16 @ 5, ReLU, max-pool 2", "32 × 16", "656"],
        ["Conv 24 @ 5, ReLU, max-pool 2", "16 × 24", "1,944"], ["Conv 32 @ 3, ReLU, global average", "32", "2,336"],
        ["Dropout 0.2, Dense 16 ReLU, Dense 1 sigmoid", "1", "545"]]
table(s, 36, 108, 410, [62, 19, 19], rows, size=11, row_h=27, align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT])
for k, (num, lab, c) in enumerate(((f"{M['params']:,}", "parameters", ORANGE),
                                   (f"{M['macs'] / 1000:.0f} k", "multiply-adds per window", SKY),
                                   (f"{M['tflite_bytes'] / 1024:.1f} KB", "full-int8 TFLite model", RED))):
    y0 = 110 + k * 66
    box(s, 466, y0, 5, 54, c)
    text(s, 480, y0, 204, 30, num, size=24, bold=True, color=NAVY)
    text(s, 480, y0 + 32, 204, 18, lab, size=12, color=MUTED)
text(s, 36, 320, 648, 90, ["Conv1D written as Conv2D with k×1 kernels, so the graph uses only six TFLite Micro ops: "
                           + ", ".join(M["ops"]) + ".",
                           f"Full-integer int8 post-training quantization; float vs int8 decisions agree on "
                           f"{pct(M['float_int8_agreement'], 2)} of test windows.",
                           "Two reference windows are embedded in the firmware and scored at every boot as a self-test."],
     size=12, bullets=True, after=5)
notes(s, "Four convolution blocks and a small dense head: 5,825 parameters, 183 thousand multiply-adds and a "
         "14 KB int8 model. Writing 1D convolutions as 2D keeps the model to six operators that TensorFlow Lite "
         "Micro supports. The firmware scores two reference windows at boot to prove the interpreter works.")

# 12 --------------------------------------------------------------- vitals
s = content_slide("4 · Design and implementation: vital signs", "Three vital signs from one optical sensor")
rows = [["Vital", "Signal chain (MAX30102 at 25 Hz)", "Alarm"],
        ["Heart rate", "IR → band-pass 0.5–4 Hz → adaptive peak detector → median beat interval", "< 40 or > 130 bpm"],
        ["SpO₂", "Ratio of ratios R = (AC/DC)red ÷ (AC/DC)IR; SpO₂ = 110 − 25·R", "< 90 %"],
        ["Breathing rate", "IR baseline, decimate 25 → 5 Hz, band-pass 0.1–0.5 Hz, count breaths", "< 8 or > 28 /min"]]
table(s, 36, 108, 648, [17, 61, 22], rows, size=11.5, row_h=34, bold_first_col=True)
card(s, 36, 260, 316, 116, "Motion gate", "When the IMU sees movement (std of |a| > 0.06 g or gyro > 35 °/s), "
                                          "PPG samples are discarded and filters re-arm, so arm swing never becomes "
                                          "a fake heartbeat.", fill=WHITE, line_color=SKY, size=12)
card(s, 368, 260, 316, 116, "30-second persistence", "An alert goes out only after a vital stays abnormal for 30 "
                                                     "consecutive valid seconds; it re-arms after 2 minutes.",
     fill=WHITE, line_color=RED, size=12)
text(s, 36, 392, 648, 30, "Filters: 4th-order Butterworth band-passes from cascaded biquads, initialised to steady "
                          "state so the large PPG DC level causes no start-up transient.", size=11, color=MUTED)
notes(s, "Heart rate uses a 0.5 to 4 hertz band-pass and peak detection. SpO2 uses the ratio of ratios. Breathing "
         "rate uses the 0.1 to 0.5 hertz band-pass from the brief on the decimated PPG baseline. The motion gate "
         "and 30-second persistence keep false alarms down.")

# 13 --------------------------------------------------------------- datasets
src = M["dataset_sources"]
s = content_slide("4 · Design and implementation: data", "Three datasets, 75 people, 37,966 windows")
rows = [["Dataset", "Sensor and placement", "People", "Trials → windows", "Role"],
        ["SisFall (Sucerquia et al., 2017)", "ADXL345 + ITG3200, waist, 200 Hz", "38 (15 aged 60+)",
         f"4,505 → {src['sisf']:,}", "Train / validate / test"],
        ["UCI Simulated Falls & ADL (Özdemir & Barshan, 2014)", "Xsens MTw, right wrist, 25 Hz", "17",
         f"3,326 → {src['uci']:,}", "12 train, 5 unseen test"],
        ["Synthetic wrist (our generator)", "Physics model of wrist falls and daily activities", "20 virtual",
         f"→ {src['syn']:,}", "Augmentation"]]
table(s, 36, 106, 648, [29, 26, 12, 16, 17], rows, size=10.5, row_h=40)
text(s, 36, 280, 648, 18, "Why each one", size=13, bold=True, color=NAVY)
text(s, 36, 302, 648, 110, [[("SisFall: ", {"bold": True}), ("the largest public fall set, 15 falls and 19 daily "
                                                              "activities, including older adults.", {})],
                            [("UCI wrist: ", {"bold": True}), ("real right-wrist recordings, the placement we build; "
                                                              "5 volunteers are never trained on.", {})],
                            [("Synthetic: ", {"bold": True}), ("covers wrist-specific motion SisFall lacks: arm swing, "
                                                              "table slaps, jumps, sit-downs.", {})]],
     size=12, bullets=True, after=5)
notes(s, "We trained on three sources. SisFall is the largest public fall dataset with 38 people including 15 older "
         "adults, but it is waist-worn. The UCI dataset gives real right-wrist recordings from 17 volunteers; we "
         "train on 12 and keep 5 as an unseen test. Our synthetic generator adds wrist-specific motions.")

# 14 --------------------------------------------------------------- data preparation
s = content_slide("4 · Design and implementation: data preparation", "From raw recordings to model-ready windows")
prep = [("Parse", "SisFall text files; UCI Xsens exports (wrist sensor only)"),
        ("Convert units", "ADC counts, m/s², rad/s → g and °/s"),
        ("Resample to 50 Hz", "SisFall 200 Hz block-average; UCI 25 Hz interpolation"),
        ("Window", "128 samples with the impact at sample 64, ±10 jitter"),
        ("Hard negatives", "ADL windows centred on each trial’s 3 strongest peaks"),
        ("Subject-wise split", "no person in both train and test"),
        ("Augment", "random 3-D rotation, gain 0.9–1.1, noise σ 0.02"),
        ("Balance", "class weights; int8 calibration on 1,000 windows")]
for k, (t_, d_) in enumerate(prep):
    col, row = k % 4, k // 4
    x0, y0 = 36 + col * 164, 108 + row * 92
    box(s, x0, y0, 156, 82, TINT, rounded=True)
    text(s, x0 + 10, y0 + 8, 24, 20, str(k + 1), size=15, bold=True, color=DARKRED)
    text(s, x0 + 32, y0 + 9, 118, 18, t_, size=12, bold=True, color=NAVY)
    text(s, x0 + 10, y0 + 32, 138, 46, d_, size=10.5, color=INK, line=1.05)
ex = np.load(ROOT / "ml_training/data/dataset.npz") if (ROOT / "ml_training/data/dataset.npz").exists() else None
cats = [f"{(i - 64) / 50:+.2f} s" for i in range(128)]
if ex is not None:
    Xd, act, srcd = ex["X"], ex["activity"], ex["source"]
    u = np.load(ROOT / "ml_training/data/uci_wrist.npz")

    def mid(Xa, m):
        idx = np.where(m)[0]
        return np.linalg.norm(Xa[idx[len(idx) // 2], :, :3], axis=1)
    series = {"Fall, waist (SisFall F01)": mid(Xd, (srcd == "sisf") & (act == "F01")),
              "Fall, wrist (UCI 901)": mid(u["X"], u["activity"] == "UCI-901"),
              "Jump, waist (SisFall D19)": mid(Xd, (srcd == "sisf") & (act == "D19"))}
    ch = line_chart(s, 36, 296, 648, 200, cats, series, [NAVY, RED, ORANGE], vmin=0, vmax=6, fmt="0", size=9,
                    widths=[1.75, 1.75, 1.75])
    ch.category_axis.tick_labels.font.size = Pt(9)
    skip_labels(ch, 16)
    ch.has_title = True
    ch.chart_title.text_frame.text = "|a| in g over one 2.56 s window (impact at 0 s)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
notes(s, "Eight preparation steps. Each trial is parsed and converted to g and degrees per second, resampled to "
         "50 hertz, and cut into 2.56-second windows with the impact at sample 64, the same alignment the "
         "firmware uses. Daily-activity windows are centred on their strongest peaks, because those are the hard "
         "cases the trigger hands to the model. Splits are by person, never by window. Training adds random "
         "rotations so the model does not depend on how the sensor sits in the strap. The chart shows a waist "
         "fall, a wrist fall and a jump.")

# 15 --------------------------------------------------------------- training
s = content_slide("4 · Design and implementation: training", "Training procedure")
rec = M["recipe"]
rows = [["Setting", "Value"], ["Training windows", f"{M['train_windows']:,} (validation: separate subjects)"],
        ["Optimiser", "Adam, learning rate 2 × 10⁻³, batch 64"],
        ["Learning-rate schedule", "× 0.5 when validation loss plateaus for 4 epochs"],
        ["Early stopping", f"patience 8; best epoch kept ({len(H['loss'])} of 40 run)"],
        ["Loss", "binary cross-entropy with balanced class weights"],
        ["Augmentation", "rotation ± 180°, gain 0.9–1.1, Gaussian noise σ 0.02"],
        ["Deployment", "int8 post-training quantization, calibrated on 1,000 windows"]]
table(s, 36, 106, 300, [38, 62], rows, size=10.5, row_h=34, bold_first_col=True)
ep = [str(i + 1) for i in range(len(H["loss"]))]
ch = line_chart(s, 352, 106, 332, 190, ep, {"train loss": H["loss"], "validation loss": H["val_loss"]},
                [NAVY, ORANGE], vmin=0, fmt="0.00", size=9)
ch.has_title = True
ch.chart_title.text_frame.text = "Loss per epoch"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
ch2 = line_chart(s, 352, 300, 332, 190, ep, {"train accuracy": H["accuracy"], "validation accuracy": H["val_accuracy"]},
                 [NAVY, ORANGE], vmin=0.85, vmax=1.0, fmt="0.00", size=9)
ch2.has_title = True
ch2.chart_title.text_frame.text = "Accuracy per epoch"
ch2.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
notes(s, "The deployed model trains with Adam, a plateau learning-rate schedule and early stopping on a "
         "validation set of separate people. Augmentation with random rotations, gain and noise makes it robust "
         "to sensor mounting and tolerance. Training and validation curves track each other closely, so the "
         "model is not over-fitting.")

# 16-21 ------------------------------------------------------------ dashboard scenarios
screenshot_slide("4 · Dashboard: caregiver view (Gradio)", "Scenario 1: normal monitoring", "01_dashboard_normal.jpg",
                 "Everything normal", "Green banner; live heart rate, SpO₂, breathing rate and battery; 15-minute trends "
                                      "built from the telemetry heartbeat.",
                 "The Gradio dashboard during normal monitoring. Vitals arrive with each telemetry heartbeat and the "
                 "trends show the last 15 minutes.")
screenshot_slide("4 · Dashboard: caregiver view (Gradio)", "Scenario 2: fall detected", "02_dashboard_fall_alert.jpg",
                 "FALL alert", "The wearer did not cancel within 15 s: a flashing red banner stays until a caregiver "
                               "acknowledges it.",
                 "A fall has been confirmed. The band sent a FALL event with the model probability and impact "
                 "strength; the banner flashes until acknowledged.")
screenshot_slide("4 · Dashboard: what the band sees (live demo)", "Scenario 3: fall and countdown",
                 "04_demo_fall_countdown.jpg", "Free fall, 5 g impact, stillness → CNN → countdown",
                 "The trace shows the free-fall dip and impact spike; the band buzzes “FALL? press A” for 15 s.",
                 "From the live demo, which runs the firmware scenario in the browser. The acceleration trace shows "
                 "the free-fall dip and the impact; the wearer lies still, the model scores it and the countdown starts.")
screenshot_slide("4 · Dashboard: what the band sees (live demo)", "Scenario 4: a jump is not a fall",
                 "05_demo_jump_rejected.jpg", "Jump rejected: “wearer recovered”",
                 "A real free fall and a 3.8 g landing, but the wearer keeps walking, so the stillness check stops it. "
                 "PPG readings are gated while moving.",
                 "A jump passes the free-fall and impact checks, but the wearer keeps moving, so it is rejected "
                 "before the model even runs. The vitals cards show GATED because the arm is moving.")
screenshot_slide("4 · Dashboard: vital signs", "Scenario 5: low blood oxygen", "06_demo_hypoxia_alert.jpg",
                 "VITALS_ABNORMAL after 30 s", "SpO₂ 87 %, heart rate 122, breathing rate 25 for over 30 consecutive "
                                              "seconds; the band shows the alarm too.",
                 "A simulated hypoxia episode. After 30 consecutive seconds below 90 percent the band sends "
                 "VITALS_ABNORMAL and the banner turns red.")
screenshot_slide("4 · Dashboard: caregiver view (Gradio)", "Scenario 6: SOS and missed medication",
                 "03_dashboard_alert_log.jpg", "Alert log", "SOS (button A held 2 s), missed doses after the confirm "
                                                           "window, abnormal vitals and falls stay open until acknowledged.",
                 "The caregiver's alert log. Critical events stay highlighted until acknowledged; confirmed doses "
                 "show in green; each row carries the vitals at the moment of the event.")

# 22 --------------------------------------------------------------- experiments overview
s = content_slide("5 · Experiments", "Seven experiments, all on unseen people")
rows = [["#", "Experiment", "Question it answers"],
        ["E1", "Baselines: threshold rules, rule-based score, Random Forest, CNN", "Is a neural network needed at all?"],
        ["E2", "Ablation: 5 training recipes", "Which data and augmentation help on a real wrist?"],
        ["E3", "5-fold subject-wise cross-validation", "Is the result stable across people?"],
        ["E4", "Decision-threshold sweep", "Where should p(fall) be cut?"],
        ["E5", "Per-activity errors", "Which falls are missed, which activities false-alarm?"],
        ["E6", "Firmware simulation, 80 simulated minutes", "Does the whole pipeline work end to end?"],
        ["E7", "Trigger analysis", "Does the threshold trigger let real falls through?"]]
table(s, 36, 106, 648, [7, 50, 43], rows, size=11, row_h=31, bold_first_col=True)
text(s, 36, 368, 648, 50, [("Test sets: ", {"bold": True}),
                           (f"{E['test_sizes']['SisFall + synthetic test (12 subj.)']:,} windows from 12 held-out SisFall "
                            f"and synthetic subjects, and {E['test_sizes']['UCI real-wrist test (5 subj.)']:,} windows from "
                            "5 held-out UCI wrist volunteers. No test person appears in any training set.", {})],
     size=12, line=1.1)
notes(s, "Seven experiments, each answering one question. The key rule: every test uses people the model never saw "
         "in training, so the numbers estimate performance on a new wearer.")

# 23 --------------------------------------------------------------- E1 baselines
s = content_slide("5 · Experiments · E1", "Baselines: the CNN wins, and stays small")
kS, kU = "SisFall + synthetic test (12 subj.)", "UCI real-wrist test (5 subj.)"
names = list(E["baselines"])
short = ["Threshold rules", "Rule-based score", "Random Forest", "CNN float32", "CNN int8 (deployed)"]
bar_chart(s, 36, 104, 430, 300, short, {"F1, SisFall + synthetic": [E["baselines"][n][kS]["f1"] for n in names],
                                        "F1, real wrist": [E["baselines"][n][kU]["f1"] for n in names]},
          [NAVY, ORANGE], vmin=0, vmax=1.05, fmt="0.0", label_fmt="0.00", size=9, major=0.2)
rows = [["Method", "Size"]] + [[sh, E["baselines_cost"][n]] for sh, n in zip(short, names)]
table(s, 478, 112, 206, [55, 45], rows, size=10, row_h=26)
text(s, 478, 282, 206, 120, ["Threshold rules alone miss most falls.", "Random Forest is close on SisFall but drops on "
                             "the real wrist and needs ~1.4 MB.", "The int8 CNN matches float accuracy at 14 KB."],
     size=11, bullets=True, after=4)
notes(s, "Threshold rules alone score an F1 around 0.25, because most real falls do not show the textbook pattern. "
         "A Random Forest on 42 hand-crafted features does well on SisFall but drops on the real wrist and would "
         "need about 1.4 megabytes. The CNN is the most accurate and fits in 14 KB after int8 quantization.")

# 24 --------------------------------------------------------------- E2 ablation
s = content_slide("5 · Experiments · E2", "Ablation: what made the real-wrist score jump")
ab = E["ablation"]
labels = ["A: SisFall only", "B: + synthetic", "C: + rotation", "D: + UCI wrist", "E: + UCI + rotation"]
bar_chart(s, 36, 104, 648, 270, labels, {"F1, SisFall + synthetic test": [ab[n][kS]["f1"] for n in ab],
                                         "F1, real-wrist test": [ab[n][kU]["f1"] for n in ab],
                                         "Recall, real-wrist test": [ab[n][kU]["recall"] for n in ab]},
          [NAVY, ORANGE, RED], vmin=0.6, vmax=1.02, fmt="0.0", label_fmt="0.000", size=9, gap=50, major=0.1)
a_, c_, e_ = (ab[k][kU]["recall"] for k in ("A  SisFall only", "C  + rotation aug.", "E  + UCI + rotation (deployed)"))
text(s, 36, 386, 648, 100, [f"Rotation augmentation alone lifts real-wrist recall from {pct(a_)} to {pct(c_)}: the "
                            "model stops depending on how the sensor is mounted.",
                            f"Adding real wrist data and rotation together gives the best wrist recall ({pct(e_)}) "
                            "with SisFall performance unchanged. This recipe is deployed."], size=12, bullets=True, after=5)
notes(s, "Five training recipes, same test people. Training on SisFall alone generalises poorly to the wrist. "
         "Random rotation augmentation is the single biggest gain, because it removes the dependence on sensor "
         "orientation. Adding real wrist data from 12 UCI volunteers plus rotation gives the best result, so that "
         "is the deployed model.")

# 25 --------------------------------------------------------------- E3 CV
s = content_slide("5 · Experiments · E3", "Five-fold cross-validation across 75 people")
cv = E["cv"]
bar_chart(s, 36, 104, 430, 300, [f"Fold {i + 1}" for i in range(len(cv["folds"]))],
          {"Accuracy": [f["accuracy"] for f in cv["folds"]], "Precision": [f["precision"] for f in cv["folds"]],
           "Recall": [f["recall"] for f in cv["folds"]]}, [NAVY, SKY, ORANGE], vmin=0.95, vmax=1.0, fmt="0.00",
          label_fmt="0.000", size=8.5, gap=60, major=0.01)
rows = [["Metric", "Mean ± std"]] + [[k.capitalize() if k != "auc" else "ROC AUC",
                                      f"{100 * cv[k + '_mean']:.2f} ± {100 * cv[k + '_std']:.2f} %"
                                      if k != "auc" else f"{cv[k + '_mean']:.4f} ± {cv[k + '_std']:.4f}"]
                                     for k in ("accuracy", "precision", "recall", "f1", "specificity", "auc")]
table(s, 478, 112, 206, [45, 55], rows, size=11, row_h=28)
text(s, 478, 320, 206, 80, "Folds split by person (GroupKFold); each fold trains a fresh model with the deployed "
                           "recipe.", size=11, color=MUTED, line=1.1)
notes(s, "To check the result is not luck of one split, we re-trained the deployed recipe five times with "
         "different held-out people. Accuracy stays around 99 percent with a standard deviation of a few tenths of "
         "a percent.")

# 26 --------------------------------------------------------------- E7 trigger
s = content_slide("5 · Experiments · E7", "Trigger analysis changed the firmware")
rows = [["Trigger rule", "Triggered falls SisFall", "Triggered falls wrist", "System recall SisFall",
         "System recall wrist"]]
for name, short_ in (("free fall <0.5 g (60 ms) then impact >2.7 g (spec)", "Free fall then > 2.7 g (spec)"),
                     ("free fall <0.6 g then impact >2.0 g", "Free fall < 0.6 g then > 2.0 g"),
                     ("impact >2.7 g only", "Impact > 2.7 g only"),
                     ("spec path OR impact >2.0 g (deployed)", "Spec OR impact > 2.0 g (deployed)")):
    r = T[name]
    rows.append([short_, pct(r["SisFall test"]["trigger_recall_falls"]), pct(r["UCI real-wrist test"]["trigger_recall_falls"]),
                 pct(r["SisFall test"]["system_recall"]), pct(r["UCI real-wrist test"]["system_recall"])])
table(s, 36, 106, 648, [32, 17, 17, 17, 17], rows, size=10.5, row_h=34, highlight_last=True,
      align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
fa = T["spec path OR impact >2.0 g (deployed)"]
text(s, 36, 290, 648, 120, [f"System recall = trigger × CNN. The spec trigger alone would cap recall near "
                            f"{pct(T['free fall <0.5 g (60 ms) then impact >2.7 g (spec)']['UCI real-wrist test']['system_recall'], 0)}.",
                            f"With the second path, recall reaches {pct(fa['SisFall test']['system_recall'])} (SisFall) and "
                            f"{pct(fa['UCI real-wrist test']['system_recall'])} (wrist).",
                            f"False alarms on the hardest daily-activity windows stay at "
                            f"{pct(fa['SisFall test']['system_false_alarm_rate_adl_peaks'])}–"
                            f"{pct(fa['UCI real-wrist test']['system_false_alarm_rate_adl_peaks'])}, before the stillness check."],
     size=12.5, bullets=True, after=6)
notes(s, "This experiment found a real design flaw before we built hardware. The CNN only runs when the threshold "
         "trigger fires. The original free-fall-then-impact rule fires for fewer than half of real falls, so the "
         "system would have missed most falls despite an excellent model. Adding an impact-only path lifts system "
         "recall to about 96 to 98 percent while false alarms stay around one percent of the hardest daily "
         "activity windows.")

# 27 --------------------------------------------------------------- results summary
s = content_slide("6 · Results", "The deployed model on 17 people it never saw")
ia = M["int8"]["all"]
big = [(pct(ia["accuracy"]), "accuracy"), (pct(ia["precision"]), "precision"), (pct(ia["recall"]), "recall"),
       (f"{ia['auc']:.3f}", "ROC AUC")]
for k, (num, lab) in enumerate(big):
    x0 = 36 + k * 164
    box(s, x0, 110, 156, 104, TINT, rounded=True)
    text(s, x0, 124, 156, 44, num, size=28, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    text(s, x0, 172, 156, 20, lab, size=12, color=MUTED, align=PP_ALIGN.CENTER)
rows = [["Test set (int8 model)", "Windows", "Accuracy", "Precision", "Recall", "F1"]]
for key, lab in (("sisf", "SisFall, 12 held-out subjects"), ("uci", "UCI real wrist, 5 held-out volunteers"),
                 ("syn", "Synthetic wrist"), ("all", "All test windows")):
    r = M["int8"][key]
    rows.append([lab, f"{r['n']:,}", pct(r["accuracy"]), pct(r["precision"]), pct(r["recall"]), f"{r['f1']:.3f}"])
table(s, 36, 232, 648, [38, 12, 12.5, 12.5, 12.5, 12.5], rows, size=11, row_h=28, highlight_last=True,
      align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 5)
text(s, 36, 384, 648, 60, [(f"Cross-validated: {100 * cv['accuracy_mean']:.1f} ± {100 * cv['accuracy_std']:.1f} % accuracy. ",
                            {"bold": True}), (f"Model {M['tflite_bytes'] / 1024:.1f} KB; float and int8 agree on "
                                              f"{pct(M['float_int8_agreement'], 2)} of decisions.", {})], size=12.5)
notes(s, "Headline results of the deployed int8 model on 9,052 windows from 17 people it never saw: about 98 percent "
         "accuracy, precision and recall, with an AUC of 0.995. Real-wrist performance is as good as SisFall.")

# 28 --------------------------------------------------------------- errors
s = content_slide("6 · Results: error analysis", "Where the remaining errors are")
per = E["per_activity"]
names_ = {"F05": "F05 trip while jogging", "F06": "F06 vertical faint", "F11": "F11 backward, sitting down",
          "F12": "F12 lateral, sitting down", "D17": "D17 getting in/out of a car", "D05": "D05 stairs, slowly",
          "D11": "D11 collapse into a chair", "D18": "D18 stumble while walking", "D19": "D19 jump to reach"}
falls = sorted((v["rate"], a) for a, v in per.items() if a.startswith("F"))[:4]
adls = sorted(((1 - v["rate"]), a) for a, v in per.items() if a.startswith("D"))[::-1][:4]
rows = [["Hardest falls (SisFall)", "Detected"]] + [[names_.get(a, a), pct(r)] for r, a in falls]
table(s, 36, 106, 318, [70, 30], rows, size=11, row_h=26)
rows = [["Activities that false-alarm most", "False alarms"]] + [[names_.get(a, a), pct(r)] for r, a in adls]
table(s, 366, 106, 318, [70, 30], rows, size=11, row_h=26)
ts = E["threshold_sweep"]
chT = line_chart(s, 36, 252, 400, 230, [f"{r['t']:.2f}" for r in ts],
                 {"Precision": [r["precision"] for r in ts], "Recall": [r["recall"] for r in ts],
                  "F1": [r["f1"] for r in ts]}, [NAVY, ORANGE, RED], vmin=0.9, vmax=1.0, fmt="0.00", size=9)
skip_labels(chT, 2)
chT.has_title = True
chT.chart_title.text_frame.text = "Decision threshold on p(fall) (deployed: 0.50)"
chT.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
text(s, 452, 262, 232, 220, ["Trip-while-jogging falls look like running until the impact; more wrist data is "
                             "the fix.", "Car entry and stumbles produce real impacts; the stillness check removes "
                             "most of these on the device.", "Threshold 0.5 sits on a flat part of the curve, so "
                             "small calibration shifts on hardware will not hurt."], size=11.5, bullets=True, after=6)
notes(s, "The few remaining misses are mostly trips while jogging, which look like running until the impact. The "
         "activities that occasionally false-alarm, like getting into a car or stumbling, involve real impacts; "
         "the stillness check removes most of these. The threshold sweep shows a flat optimum around 0.5.")

# 29 --------------------------------------------------------------- firmware simulation
s = content_slide("6 · Results: firmware", "The firmware pipeline, end to end, in simulation")
for k, (num, lab) in enumerate(((f"{FW['falls_detected']}/{FW['falls_total']}", "scripted falls detected"),
                                (f"{FW['false_triggers']}/{FW['hard_negatives']}", "false alerts from jumps and slaps"),
                                (f"{FW['tests']}/{FW['tests']}", "unit tests passing"))):
    x0 = 36 + k * 220
    box(s, x0, 110, 208, 92, TINT, rounded=True)
    text(s, x0, 120, 208, 40, num, size=28, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    text(s, x0, 164, 208, 30, lab, size=12, color=MUTED, align=PP_ALIGN.CENTER)
rows = [["Vital sign (vs scenario ground truth)", "Mean abs. error", "RMSE"],
        ["Heart rate", f"{FW['hr_mae']:.1f} bpm", f"{FW['hr_rmse']:.1f} bpm"],
        ["SpO₂", f"{FW['spo2_mae']:.1f} %", f"{FW['spo2_rmse']:.1f} %"],
        ["Breathing rate", f"{FW['rr_mae']:.1f} /min", f"{FW['rr_rmse']:.1f} /min"]]
table(s, 36, 218, 400, [50, 25, 25], rows, size=11, row_h=27, align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
text(s, 452, 218, 232, 150, [f"{FW['minutes']} simulated minutes ({FW['cycles']} scenario cycles) through the real "
                             "C++ firmware modules on a PC.", "Every fall window scored p = 0.996 by the int8 model.",
                             "RMSE is higher than MAE because estimates lag sudden step changes by a few seconds."],
     size=11.5, bullets=True, after=5)
text(s, 36, 352, 648, 50, [("Build: ", {"bold": True}), (FW["build"], {})], size=11.5, color=INK)
notes(s, "We compiled the actual firmware modules on a PC and replayed the simulation scenario for 80 minutes. "
         "Every scripted fall was detected, no jump or table slap caused an alert, and vital signs tracked ground "
         "truth within a few units. All unit tests pass and the firmware compiles for the ESP32 in all build modes.")

# 30 --------------------------------------------------------------- feasibility
s = content_slide("6 · Results: hardware feasibility", "Does the software fit the hardware?")
rows = [["Resource", "Budget", "Needed", "Status"],
        ["App flash", "1.9 MB slot", FW["flash"], "Fits (measured)"],
        ["Static RAM", "320 KB", FW["ram"], "Fits (measured)"],
        ["Model + tensor arena", "—", f"{M['tflite_bytes'] / 1024:.1f} KB + 24 KB", "Fits; usage logged at boot"],
        ["Inference time", "< 20 ms (one sample)", f"{M['macs'] / 1000:.0f} k MACs ≈ 5–15 ms (est.)", "To measure on device"],
        ["Battery, 500 mAh", "≥ 24 h target", "≈ 40–55 mA → 9–12 h (est.)", "Needs power work"],
        ["Radio", "Off when idle", "≈ 2 s per 5 min + events", "Implemented"]]
table(s, 36, 106, 648, [22, 22, 32, 24], rows, size=11, row_h=30, bold_first_col=True)
text(s, 36, 330, 648, 90, ["Measured: compiled firmware size and memory for the ESP32.",
                           "Estimated: inference time from the multiply-add count; current from datasheet typicals "
                           "(ESP32 active ≈ 30–50 mA, MPU6050 ≈ 4 mA, MAX30102 ≈ 1–2 mA).",
                           "The firmware prints real inference time, arena use and stack margins at boot, so week 1 "
                           "of the build replaces every estimate with a measurement."], size=11.5, bullets=True, after=5)
notes(s, "Memory is comfortably within budget: about half the flash slot and a quarter of RAM. Inference time "
         "and battery life are estimates; the firmware logs the real numbers at boot so we can confirm them in "
         "the first week of the hardware build. Battery life is the main open risk.")

# 31 --------------------------------------------------------------- readiness
s = content_slide("6 · Results", "Is the software ready for the hardware?")
cols = [("Done and verified", NAVY, ["Firmware compiles in all 3 build modes", f"{FW['tests']} unit tests pass",
                                     f"{FW['falls_detected']}/{FW['falls_total']} falls, 0 false alerts in simulation",
                                     "Model: 98 % on 17 unseen people", "Dashboard + webhook tested end to end"]),
        ("Changed for hardware", RED, ["Impact-only trigger path (recall 43 % → 97 %)",
                                       "Rotation-trained model with real wrist data", "Task watchdog and I²C bus scan",
                                       "IMU axis remap, wrist LED current", "Data-logger build + recorder script"]),
        ("Needs the hardware", ORANGE, ["Inference time and arena on ESP32", "Sensor orientation on the wrist",
                                        "PPG quality vs a reference oximeter", "Battery life, WiFi range",
                                        "Fine-tuning on our own wrist data"])]
for k, (t_, c_, items_) in enumerate(cols):
    x0 = 36 + k * 220
    box(s, x0, 108, 208, 34, c_, rounded=True)
    text(s, x0 + 12, 108, 184, 34, t_, size=13, bold=True, color=WHITE if c_ != ORANGE else NAVY,
         anchor=MSO_ANCHOR.MIDDLE)
    box(s, x0, 148, 208, 250, TINT, rounded=True)
    text(s, x0 + 10, 158, 188, 236, items_, size=11.5, bullets=True, after=7)
text(s, 36, 412, 648, 40, [("Verdict: ", {"bold": True}), ("the software is complete and ready to flash; what remains "
                                                          "can only be measured on the physical wristband.", {})],
     size=13, color=NAVY)
notes(s, "The software is complete. Before building hardware we made five changes: the extra trigger path, a "
         "rotation-trained model with real wrist data, a watchdog and bus scan, an axis remap and LED setting for "
         "the wrist, and a data-logger build. What remains can only be measured on the real device.")

# 32 --------------------------------------------------------------- plan
s = content_slide("7 · Future work: plan of action", "From software to a working wristband by 10 December")
rows = [["Week (2026)", "Tasks", "Exit criterion"],
        ["1 · 12–18 Nov", "Order parts; breadboard ESP32 + MPU6050 + MAX30102 + DS3231; flash the hardware build",
         "I²C scan finds all 4 chips; self-test passes; inference time logged"],
        ["2 · 19–25 Nov", "Data-logger build: record wrist trials from 5 volunteers (falls onto crash mats); set "
                          "IMU axis map; tune PPG LED vs a reference oximeter",
         "≥ 300 labelled trials; SpO₂ within ± 2 % of reference"],
        ["3 · 26 Nov–2 Dec", "Fine-tune the CNN on our wrist data; solder the board; 3D-print case and strap; fit "
                             "battery; measure current", "Wearable prototype v1; model re-validated"],
        ["4 · 3–9 Dec", "System tests: 50 mat falls, 2-hour wear test, vitals vs references, battery run-down, "
                        "dashboard alerts", "Recall ≥ 95 %, ≤ 1 false alarm per 2 h; test report"],
        ["10 Dec", "Final demonstration and report submission", "Working wristband alerting the dashboard"]]
table(s, 36, 104, 648, [16, 50, 34], rows, size=10.5, row_h=56, highlight_last=True, bold_first_col=True)
text(s, 36, 456, 648, 30, "Falls are performed only by healthy volunteers onto crash mats. Task owners: [team member names].",
     size=10, color=MUTED)
notes(s, "Week one: parts and breadboard; the I2C scan and self-test confirm the hardware. Week two: record real "
         "wrist data and calibrate. Week three: fine-tune the model and build the wearable. Week four: formal tests "
         "with clear pass criteria. Final demonstration on 10 December.")

# 33 --------------------------------------------------------------- risks and after
s = content_slide("7 · Future work: risks and next steps", "Risks we expect, and what comes after")
rows = [["Risk", "Mitigation"], ["Battery life (est. 9–12 h)", "CPU at 80 MHz, light sleep, MPU FIFO batching; ≥ 24 h"],
        ["Wrist PPG is noisier than a fingertip", "LED tuning, motion gate, report only valid readings"],
        ["Sensor mounted at another angle", "Rotation-trained model + IMU_AXIS_MAP"],
        ["Dev board too large to wear", "ESP32-S3 mini module on a custom PCB"],
        ["Realistic fall data is scarce", "Volunteers on crash mats; data-logger ready"]]
table(s, 36, 106, 400, [42, 58], rows, size=10.5, row_h=40)
text(s, 456, 106, 228, 18, "After 10 December", size=14, bold=True, color=NAVY)
text(s, 456, 132, 228, 260, ["BLE link and a phone app for caregivers", "LTE-M or phone tethering for alerts outdoors",
                             "Over-the-air firmware updates", "TLS and per-device keys on the webhook",
                             "Longer trials with older adults, with ethics approval"], size=12, bullets=True, after=8)
notes(s, "Battery life is the biggest risk; we plan to lower the clock, sleep between samples and batch IMU "
         "readings. After the course: a phone app, outdoor connectivity, OTA updates, encryption and properly "
         "approved trials with older adults.")

# 34 --------------------------------------------------------------- thank you (template title style)
_n[0] += 1
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(str(ASSETS / "title_bg.png"), 0, 0, Pt(720), Pt(540))
box(s, 0, 264, 684, 216, NAVY)
for x0, x1, c in ((0, 228, ORANGE), (228, 456, SKY), (456, 684, RED)):
    box(s, x0, 480, x1 - x0, 6, c)
s.shapes.add_picture(str(ASSETS / "crest.png"), Pt(22), Pt(276), Pt(96), Pt(92))
text(s, 12, 372, 150, 32, [("BITS", {"bold": True}), (" Pilani", {})], size=27, color=WHITE)
text(s, 14, 404, 140, 14, "Dubai Campus", size=11, color=WHITE)
text(s, 160, 278, 510, 50, "Thank you", size=36, bold=True, color=WHITE)
text(s, 160, 334, 510, 20, "Questions are welcome.", size=15, color=PALE)
text(s, 160, 370, 510, 90, [[("Live demo: ", {"bold": True}), ("huggingface.co/spaces/kkshik/vital-wristband", {})],
                            [("Source code: ", {"bold": True}), ("github.com/f20230104-dotcom/Vital-armband", {})]],
     size=13, color=WHITE, after=8)
notes(s, "Thank you. The live demo is on Hugging Face and all source code is on GitHub.")

prs.save(OUT)
print(f"wrote {OUT} ({len(prs.slides)} slides)")
