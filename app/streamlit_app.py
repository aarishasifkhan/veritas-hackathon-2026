"""BleedCheck CT: a research prototype. Not medical advice."""
import base64
import io
import threading
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms as T

# ------------------------------------------------------------------ things you may need to edit
GITHUB_USER = "aarishasifkhan"
GITHUB_REPO = "veritas-hackathon-2026"
WEIGHTS_VERSION = "v1"                    # the release tag holding the weights; use "v2" if you publish new weights later
# The release asset can be named any of these. The app tries each in turn, so a ".pt.zip" or ".zip" upload also works
# (a PyTorch .pt file is itself a zip archive, and torch.load reads it by content, not by file name).
WEIGHT_FILE_NAMES = ["bleedcheck_weights.pt", "bleedcheck_weights.pt.zip", "bleedcheck_weights.zip"]
WEIGHTS_URLS = [f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{WEIGHTS_VERSION}/{n}" for n in WEIGHT_FILE_NAMES]
REPO_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"
NOTEBOOK_URL = "https://www.kaggle.com/code/aarishasifkhan/bleedcheck-ct-a-second-look-at-head-ct-scans"
VIDEO_URL = "https://youtu.be/ZO6M_Wn9Gow"

# Copied from the README. Columns: test, bleeds caught, false alarms, AUC.
# The first three rows are the mean (and range) of four training runs. The last row is the demo model alone.
RESULTS_ROWS = [
    ("Baseline: always say \"no bleed\"", "0%", "0%", "n/a"),
    ("Same source as the training data (998 slices, 164 bleeds)", "92.1% (82.9–97.6%)", "1.7% (0.6–2.8%)", "0.996 (0.991–0.998)"),
    ("A different public source, cut-off 0.5 (50 bleeds, 50 normal)", "98.5% (96–100%)", "27.5% (16–48%)", "0.946 (0.893–0.985)"),
    ("Different source, after tuning on 100 local scans (demo model only)", "96.0%", "2.0%", "0.999"),
]

# How many local scans does a new hospital need? Demo model, 3 random draws each, tested on 100 scans it never saw.
# Columns: local scans, false alarms (new source), bleeds caught (new source), AUC (new source), bleeds caught (original data).
TUNING_ROWS = [
    ("0 (no tuning)", "16.0%", "100%", "0.985", "97.0%"),
    ("10", "4.0%", "90.7%", "0.989", "88.8%"),
    ("20", "6.0%", "95.3%", "0.988", "89.0%"),
    ("40", "1.3%", "95.3%", "0.998", "86.4%"),
    ("100", "2.0%", "96.0%", "0.999", "90.9%"),
]

HERE = Path(__file__).parent
WEIGHTS = HERE / f"bleedcheck_weights_{WEIGHTS_VERSION}.pt"
EXAMPLES = HERE
BRAIN_IMAGE = HERE / "assets" / "brain.png"
CLASSES = ["Normal", "Bleeding", "Ischemia"]
SHOW_ORDER = [1, 2, 0]                    # show Bleeding first, then Ischemia, then Normal

torch.set_num_threads(2)   # shared hosts report many CPU cores but give you ~2; using them all slows PyTorch down

st.set_page_config(page_title="BleedCheck CT", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ look and feel
# Cream paper, steel blue, neon pink and yellow accents, and red.
# Red means one thing: "possible bleed".
CSS = """
:root { --paper:#FFFFF0; --steel:#4A7B9D; --ink:#1F3F57; --alert:#CC0000; }
html, body, [data-testid="stAppViewContainer"], .stApp { background: var(--paper); color: var(--ink); }
.stApp { font-family: 'Atkinson Hyperlegible', system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 1.0625rem; line-height: 1.6; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 70rem; padding: 1.6rem 1.5rem 3.5rem; }

.stApp a { color: var(--ink); text-decoration-thickness: 2px; text-underline-offset: 3px; }
.stApp a:hover { color: var(--steel); }
.stApp :focus-visible { outline: 3px solid var(--ink) !important; outline-offset: 3px !important; box-shadow: 0 0 0 6px var(--paper), 0 0 0 9px var(--steel) !important; }

.stApp .bc-title, .stApp .bc-h2, .bc-brand { font-family: 'Newsreader', Georgia, 'Times New Roman', serif; color: var(--ink); font-weight: 600; letter-spacing: -.01em; }
.topbar { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding-bottom: 12px; border-bottom: 2px solid var(--steel); margin-bottom: 30px; }
.bc-brand { font-size: 1.4rem; }
.tag { font-size: .95rem; }

.hero { border-left: 6px solid var(--steel); padding: 2px 0 2px 20px; margin-bottom: 22px; }
.stApp h1.bc-title { font-size: clamp(2.1rem, 5vw, 3.3rem); line-height: 1.1; margin: 0 0 12px; padding: 0; }
.hero p { margin: 6px 0; max-width: 62ch; }
.hero .lede { font-size: 1.2rem; }
.notice { border: 2px solid var(--steel); border-radius: 8px; padding: 12px 16px; margin: 0 0 34px; max-width: 70ch; }
.notice p { margin: 4px 0; }

.stApp h2.bc-h2 { font-size: 1.55rem; line-height: 1.2; margin: 0 0 10px; padding: 0; }
.hint { font-size: .95rem; margin: 8px 0 14px; }
.empty { border: 2px dashed var(--steel); border-radius: 8px; padding: 26px 22px; }
.empty b { font-family: 'Newsreader', Georgia, serif; font-size: 1.25rem; }

/* The brain sits inside a CT-scanner ring. The ring turns red when the tool flags a scan. */
.stage { margin: 0; padding: 22px 0 0; text-align: center; }
.gantry { position: relative; width: min(88%, 24rem); aspect-ratio: 1 / 1; margin: 0 auto; border-radius: 50%; border: 6px solid var(--steel); display: grid; place-items: center; }
.brain-art { width: 82%; aspect-ratio: 640 / 531; background-repeat: no-repeat; background-position: center; background-size: contain; position: relative; z-index: 1; }
.gantry.flag { border-color: var(--alert); }
.stage figcaption { font-size: .92rem; margin-top: 34px; max-width: 30ch; margin-left: auto; margin-right: auto; }

.verdict { display: flex; gap: 14px; align-items: flex-start; padding: 14px 18px; border: 2px solid var(--steel); border-left-width: 12px; border-radius: 8px; margin-bottom: 14px; }
.verdict svg { flex: none; width: 2.2rem; height: 2.2rem; margin-top: 2px; }
.v-title { font-family: 'Newsreader', Georgia, serif; font-size: 1.6rem; font-weight: 600; line-height: 1.15; display: block; }
.verdict p { margin: 4px 0 0; }
.verdict.flag { border-color: var(--alert); }
.verdict.flag .v-title { color: var(--alert); }

.findings { border: 2px solid var(--steel); border-radius: 8px; padding: 6px 16px 12px; }
.bc-row { display: grid; grid-template-columns: 6.5rem 1fr 3.6rem; gap: 12px; align-items: center; margin: 12px 0; }
.bc-track { position: relative; height: 1.3rem; border: 2px solid var(--steel); border-radius: 3px; background: var(--paper); }
.bc-fill { height: 100%; background: var(--steel); }
.bc-fill.hot { background: var(--alert); }
.bc-line { position: absolute; top: -8px; bottom: -8px; width: 3px; background: var(--ink); }
.bc-num { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
.caption { font-size: .95rem; line-height: 1.5; margin: 6px 0 0; }

.section { margin-top: 44px; }
.prose { max-width: 78ch; margin: 0 0 14px; }
.prose p { margin: 6px 0; }
.prose ul { margin: 6px 0 0 1.2rem; padding: 0; }
.prose li { margin: 4px 0; }
.results { width: 100%; border-collapse: collapse; border: 2px solid var(--ink); font-size: 1rem; }
.results caption { text-align: left; padding: 0 0 8px; font-size: .95rem; caption-side: bottom; }
.results th { text-align: left; font-weight: 700; padding: 10px 14px; background: var(--ink); color: var(--paper); }
.results td { padding: 11px 14px; border-bottom: 1px solid var(--steel); }
.results tr:last-child td { border-bottom: none; }
.results td:not(:first-child), .results th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; width: 20%; }
.results.tune td:not(:first-child), .results.tune th:not(:first-child) { width: 17%; }
.results tr.base td { font-style: italic; border-bottom: 2px solid var(--ink); }

.limits { background: var(--ink); color: var(--paper); border-radius: 8px; padding: 16px 22px; margin: 26px 0 0; max-width: 78ch; }
.limits h3 { font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; margin: 0 0 6px; padding: 0; color: var(--paper); }
.limits p { margin: 6px 0; color: var(--paper); }
.footer { margin-top: 40px; padding-top: 16px; border-top: 2px solid var(--steel); font-size: .95rem; line-height: 1.65; max-width: 80ch; }

/* Streamlit widgets */
[data-testid="stFileUploaderDropzone"] { background: var(--paper); border: 2px dashed var(--steel); border-radius: 8px; color: var(--ink); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--ink); }
[data-testid="stWidgetLabel"] p { font-weight: 700; color: var(--ink); }
[data-baseweb="select"] > div { background: var(--paper); border: 2px solid var(--steel); color: var(--ink); }
.stButton > button, [data-testid="stFileUploaderDropzone"] button { min-height: 2.75rem; border-radius: 6px; border: 2px solid var(--ink); background: var(--ink); color: var(--paper); font-weight: 700; }
.stButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover { background: var(--steel); border-color: var(--steel); color: var(--paper); }
.stButton > button *, [data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: var(--paper) !important; }
[data-testid="stExpander"] { border: 2px solid var(--steel); border-radius: 8px; background: transparent; }
[data-testid="stImage"] img { border: 2px solid var(--ink); border-radius: 4px; }
[data-testid="stImageCaption"] { color: var(--ink); font-size: .95rem; }

@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1rem .9rem 2.5rem; }
  .topbar { flex-direction: column; gap: 2px; }
  .bc-row { grid-template-columns: 5.2rem 1fr 3.4rem; gap: 8px; }
  .results { font-size: .85rem; } .results th, .results td { padding: 8px 6px; }
}
/* ================= NEON YELLOW + PINK, ONLY IN A FEW PLACES ================= */
:root { --pink:#FF2E9A; --yellow:#F5FF1F; }
/* --- Brain: pink aura, gradient ring --- */
.gantry:not(.flag) {
  border-color: transparent;
  background:
    radial-gradient(circle, rgba(255,46,154,.20) 0%, rgba(255,46,154,.08) 55%, transparent 72%) padding-box,
    linear-gradient(var(--paper), var(--paper)) padding-box,
    conic-gradient(var(--pink), var(--yellow), var(--pink), var(--yellow), var(--pink)) border-box;
  box-shadow: 0 0 22px 3px rgba(255,46,154,.45), 0 0 46px 8px rgba(245,255,31,.25);
}
.gantry.flag { box-shadow: 0 0 18px 3px rgba(204,0,0,.45); }
.brain-art { filter: drop-shadow(0 0 10px rgba(255,46,154,.65)); }

/* --- Top bar: gradient line --- */
.topbar { position: relative; border-bottom: none; }
.topbar::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px; border-radius: 2px;
  background: linear-gradient(90deg, var(--pink), var(--yellow), var(--pink), var(--yellow), var(--pink)); background-size: 200% 100%; }

/* --- Title bar: gradient strip beside the intro --- */
.hero { position: relative; border-left: none; padding-left: 26px; }
.hero::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; border-radius: 3px;
  background: linear-gradient(180deg, var(--pink), var(--yellow), var(--pink), var(--yellow), var(--pink)); background-size: 100% 200%; }
/* --- Heading underlines --- */
.stApp h2.bc-h2::after { content: ""; display: block; width: 3.6rem; height: 4px; margin-top: 7px; border-radius: 2px;
  background: linear-gradient(90deg, var(--pink), var(--yellow), var(--pink), var(--yellow), var(--pink)); background-size: 200% 100%; }

/* --- Limits box: gradient border, gradient title --- */
.limits { border: 3px solid transparent;
  background: linear-gradient(var(--ink), var(--ink)) padding-box, linear-gradient(110deg, var(--pink), var(--yellow), var(--pink), var(--yellow), var(--pink)) border-box;
  background-size: 100% 100%, 100% 100%; }
.limits h3 { background: linear-gradient(90deg, var(--pink), var(--yellow), var(--pink)); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; display: inline-block; }

/* --- Footer line --- */
.footer { position: relative; border-top: none; }
.footer::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; border-radius: 2px;
  background: linear-gradient(90deg, var(--pink), var(--yellow), var(--pink), var(--yellow), var(--pink)); background-size: 200% 100%; }

.loading { aspect-ratio: 1 / 1; display: grid; place-items: center; text-align: center; padding: 12px; border: 2px dashed var(--steel); border-radius: 4px; background: rgba(74,123,157,.08); }
"""

FONT_CSS = "@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');"

def _clean(html):
    # Streamlit's markdown breaks on blank or indented lines inside HTML, so squash them
    return " ".join(line.strip() for line in html.splitlines() if line.strip())

CSS_CLEAN = _clean(CSS)

CHECK_SVG = ('<svg viewBox="0 0 32 32" aria-hidden="true" focusable="false"><circle cx="16" cy="16" r="14" fill="none" stroke="#1F3F57" stroke-width="2.5"/>'
             '<path d="M9 16.5l5 5 9-11" fill="none" stroke="#1F3F57" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')
FLAG_SVG = ('<svg viewBox="0 0 32 32" aria-hidden="true" focusable="false"><path d="M16 3L30 28H2z" fill="#CC0000"/>'
            '<path d="M16 12v8" stroke="#FFFFF0" stroke-width="3" stroke-linecap="round"/><circle cx="16" cy="24" r="1.8" fill="#FFFFF0"/></svg>')

TOPBAR_HTML = '<div class="topbar"><span class="bc-brand">BleedCheck CT</span><span class="tag">A student prototype for the Veritas AI Hackathon 2026</span></div>'

INTRO_HTML = """
<div class="hero">
<h1 class="bc-title">A second look at head CT scans</h1>
<p class="lede">Upload one head CT slice. BleedCheck estimates the chance of a brain bleed and shows what drove that estimate, so an emergency doctor can decide which scans to open first.</p>
<p>We built it for the doctor on a night shift with no radiologist in the building. In our tests it caught 92 of every 100 bleeds, and it missed about 1 in 13.</p>
</div>
<div class="notice">
<p><b>This project is a prototype and is not medical advice.</b></p>
<p>The tool does not diagnose anyone. Please do not upload a real patient's scan. The app processes images in memory and does not store them. The free host can sleep when unused, so the first load may take a minute.</p>
</div>
"""

EMPTY_HTML = """
<div class="empty"><b>No scan selected yet.</b><br>Upload an image on the left, or choose a public example. The result and the heatmap appear here.</div>
"""

LOADING_HTML = '<div class="loading" role="status">Drawing the heatmap...</div>'

BRAIN_ALT = "Side view of a human brain, shaded in steel blue and cream. This is an illustration, not your scan."
BRAIN_PATHS = [HERE / "assets" / "brain.png", HERE / "brain.png", HERE.parent / "assets" / "brain.png",
               HERE.parent / "app" / "assets" / "brain.png"]

@st.cache_data
def _read_b64(path_str):
    # Shrink the brain picture once and send it as WebP: the page re-sends it on every interaction, so smaller is faster
    raw = Path(path_str).read_bytes()
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        im.thumbnail((640, 640))
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=85, method=6)
        return "image/webp", base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return "image/png", base64.b64encode(raw).decode()

def brain_b64():
    for p in BRAIN_PATHS:
        if p.exists():
            return _read_b64(str(p))
    return "", ""   # not cached, so adding the file later works without a reboot

def stage_html(flagged):
    mime, b64 = brain_b64()
    # The image goes in CSS (a data URI in an <img> tag can be stripped by Streamlit's markdown renderer).
    css = f'<style>.brain-art {{ background-image: url("data:{mime};base64,{b64}"); }}</style>' if b64 else ""
    art = f'<div class="brain-art" role="img" aria-label="{BRAIN_ALT}"></div>' if b64 else ""
    cls = "gantry flag" if flagged else "gantry"
    note = ("An illustration of a human brain, drawn for this project. It is not your scan." if b64
            else "Brain image not found. Add app/assets/brain.png to the repository.")
    return css + _clean(f'<figure class="stage"><div class="{cls}">{art}</div><figcaption>{note}</figcaption></figure>')

def verdict_html(p, thr):
    flagged = p >= thr
    cls, icon = ("flag", FLAG_SVG) if flagged else ("clear", CHECK_SVG)
    title = "Possible bleed" if flagged else "No bleed flagged"
    text = ("The bleeding probability is at or above the alert line. Treat this as a reason to open the scan first, not as a diagnosis."
            if flagged else "The bleeding probability is below the alert line. This does not rule out a bleed.")
    return _clean(f"""<div class="verdict {cls}" role="status" aria-live="polite">{icon}
    <div><span class="v-title">{title}</span><p>{text}</p></div></div>""")

def findings_html(probs, thr):
    rows = []
    for i in SHOW_ORDER:
        name, pct = CLASSES[i], min(max(probs[i], 0.0), 1.0) * 100
        is_bleed = i == 1
        hot = " hot" if is_bleed and probs[i] >= thr else ""
        line = f'<span class="bc-line" style="left:{thr * 100:.0f}%"></span>' if is_bleed else ""
        rows.append(f'<div class="bc-row"><span>{name}</span><div class="bc-track" role="img" aria-label="{name}: {pct:.0f} percent">'
                    f'<div class="bc-fill{hot}" style="width:{pct:.1f}%"></div>{line}</div><span class="bc-num">{pct:.0f}%</span></div>')
    note = f'<p class="caption">The black line is the alert line. A bleeding probability of {thr * 100:.0f}% or higher raises the flag.</p>'
    return _clean(f'<div class="findings">{"".join(rows)}{note}</div>')

def table_html(caption, headers, rows, first_row_base=False, extra_class=""):
    body = ""
    for n, row in enumerate(rows):
        tr = '<tr class="base">' if (first_row_base and n == 0) else "<tr>"
        body += tr + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    head = "".join(f'<th scope="col">{h}</th>' for h in headers)
    cls = f"results {extra_class}".strip()
    return _clean(f'<table class="{cls}"><caption>{caption}</caption><tr>{head}</tr>{body}</table>')

def results_html(rows):
    return table_html(
        "A false alarm is a normal or ischemia scan that the model flagged as a possible bleed. "
        "The first three rows show the mean and, in brackets, the lowest and highest of four training runs. "
        "The last row is the demo model alone, tested on 100 scans it never trained on. "
        "An always-\"no bleed\" model is 83.6% accurate yet catches no bleeds, so we do not rely on accuracy. "
        "Same-source scores are an upper bound because the data has no patient IDs.",
        ["Test", "Bleeds caught", "False alarms", "AUC"], rows, first_row_base=True)

def tuning_html(rows):
    return table_html(
        "Demo model, tuned on labelled scans from the first half of the new source and tested on the other 100 scans (3 random draws each).",
        ["Local scans", "False alarms (new source)", "Bleeds caught (new source)", "AUC (new source)", "Bleeds caught (original data)"],
        rows, extra_class="tune")

HOW_HTML = """
<div class="prose">
<p>BleedCheck CT reads one head CT slice and returns three probabilities: bleeding, ischemia and normal. It shows "Possible bleed" when the bleeding probability is 50% or higher, and "No bleed flagged" otherwise. Ischemia counts as "no bleed", so the model has to tell a bleed from a clot, not only from a healthy brain.</p>
<p>We fine-tuned a ResNet18 pretrained on ImageNet on the Brain Stroke CT Dataset: 6,650 anonymised slices (4,427 normal, 1,093 bleeding, 1,130 ischemia), split 70/15/15. We tested on 200 scans from a different public source, using half to compare experiments and fine-tune and keeping half for one final test. We trained every recipe with four random seeds, because a single run can mislead. In our first attempt one run flagged 59% of the new source's normal scans and a re-run of the same recipe flagged 16%.</p>
<p>We picked the recipe with a rule we wrote before looking at the test split or the final half of the new source. Augmentation lowered false alarms on the new source, but it also lowered bleeds caught on the original data and roughly tripled false alarms there, so the rule chose the plain ResNet18.</p>
</div>
"""

TUNING_INTRO_HTML = """
<div class="prose"><p>Our own question: does the tool still work on scans from a different source, and how many local scans does it need to adapt?</p></div>
"""

TUNING_NOTE_HTML = """
<div class="prose">
<p>With about 40 to 100 local scans, false alarms on the new source fell from 16% to about 1 to 2%, and even 10 scans cut them to 4%. This came with costs:</p>
<ul>
<li>Bleeds caught on the new source dropped from 100% to about 91 to 96%, so the model missed a few more bleeds as it became less trigger-happy.</li>
<li>The model got worse on the original data. Bleeds caught there fell from 97.0% to 86.4% at 40 scans and recovered only to 90.9% at 100 scans. At 100 scans, false alarms on the original data also rose from 1.3% to 4.7%.</li>
</ul>
<p>The 0-scan row is the single demo model, which happens to be the best of the four runs on the new source, so its starting point looks better than the average in the table above. At 100 local scans the model sees every scan in the local half, so the three draws differ only in training order. This is one demo model, one new source and three draws, so treat it as a lead, not a guarantee. A hospital should test the tool on its own labelled scans before relying on it.</p>
</div>
"""

LIMITS_HTML = """
<aside class="limits"><h3>Read the limits before you trust a result</h3>
<p>On scans like its training data, the model missed about 1 in 13 bleeds. Across four training runs, the miss rate ranged from 2% to 17%, so one unlucky run can be much worse than another.</p>
<p>On scans from a different public source, it wrongly flagged 16 to 48% of normal scans (27.5% on average) at the 0.5 alert line, and its ranking quality (AUC) dropped from 0.996 to 0.946. Moving the alert line did not fix this reliably: it helped some runs and hurt others. Tuning on 40 to 100 labelled local scans cut false alarms to about 1 to 2%, but the model then caught fewer bleeds and did worse on the original data.</p>
<p>The heatmap shows what drove the score, not where a bleed is. On missed bleeds it often lit up the back of the head or the head-holder, and on some normal scans the image corners and text overlays. The model may lean on cues that are not the bleed.</p>
<p>Our data sets are small and public. They have no patient IDs, so slices from one patient can land in both training and test and same-source scores are an upper bound. The new-source test holds only 100 scans from one publisher. Treat the numbers as a study, not a guarantee.</p></aside>
"""

NEXT_HTML = """
<div class="prose">
<ul>
<li>Use data with patient IDs, so we can split by patient and remove the leakage risk.</li>
<li>Test scans from more hospitals, and set the alert cut-off on each hospital's own scans.</li>
<li>Mask the skull and image background so the model cannot lean on them.</li>
<li>Check performance by bleed size, patient age and sex, once the data allows it.</li>
<li>Test with clinicians.</li>
</ul>
</div>
"""

FOOTER_HTML = f"""
<div class="footer"><b>This project is a prototype and is not medical advice.</b><br>
Made by Aarish Asif Khan and Aabir Asif Khan, Islamabad. Code: <a href="{REPO_URL}">GitHub</a>. Notebook with outputs: <a href="{NOTEBOOK_URL}">Kaggle</a>. Submission video: <a href="{VIDEO_URL}">YouTube</a>.<br>
We trained on the Brain Stroke CT Dataset, a copy of the TEKNOFEST-2021 stroke data set from the Republic of Türkiye Ministry of Health (Koç et al., Eurasian J Med 2022).
We tested on Head CT - hemorrhage by Felipe Kitamura (CC0). The example scans come from the CC0 set. We built the app with PyTorch, torchvision and Streamlit.<br>
AI tools: Claude (Anthropic) was our main AI assistant. It helped us plan, explained concepts, and wrote and debugged much of the code, which we ran, checked and adjusted ourselves. ChatGPT was used only to brainstorm project names. We ran every experiment and made the decisions. No other AI tools were used.</div>
"""

# ------------------------------------------------------------------ the model
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
LOCK = threading.Lock()      # the cached model is shared, so run one analysis at a time
CREAM = np.array([1.0, 1.0, 0.94], dtype=np.float32)

def _download_weights():
    # Download to a temporary file first, so an interrupted download never leaves a broken .pt behind.
    # Try each possible release file name until one works.
    tmp = WEIGHTS.with_suffix(".tmp")
    errors = []
    try:
        for url in WEIGHTS_URLS:
            try:
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(WEIGHTS)
                return
            except Exception as e:
                errors.append(f"{url} -> {e}")
        raise RuntimeError("Could not download the model weights. Tried:\n" + "\n".join(errors))
    finally:
        if tmp.exists():
            tmp.unlink()

@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def load_model():
    if not WEIGHTS.exists():
        _download_weights()
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    try:
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    except Exception:
        # A corrupt or wrong file (for example a Git LFS pointer): delete it so the next run downloads a fresh copy
        WEIGHTS.unlink(missing_ok=True)
        raise
    return m.eval()

def gradcam(model, x, class_idx=1):
    feats, grads = [], []
    def fwd(m, i, o):
        feats.append(o)
        o.register_hook(lambda g: grads.append(g))
    h = model.layer4.register_forward_hook(fwd)
    model.zero_grad()
    out = model(x)
    out[0, class_idx].backward()
    h.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
    return cam, out.softmax(1)[0].detach().cpu().numpy()

@st.cache_data(show_spinner="Analysing the scan...", max_entries=64)
def predict(data: bytes):
    """Fast part: one forward pass. This is all we need to show the verdict."""
    model = load_model()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = TF(img).unsqueeze(0)
    with LOCK, torch.inference_mode():
        p = model(x).softmax(1)[0].cpu().numpy()
    grey = np.asarray(img.convert("L").resize((448, 448)), dtype=np.float32) / 255
    return [float(v) for v in p], (grey * 255).astype("uint8")

@st.cache_data(show_spinner=False, max_entries=64)
def heatmap(data: bytes):
    """Slower part: Grad-CAM needs a backward pass, so it runs after the verdict is already on screen."""
    model = load_model()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = TF(img).unsqueeze(0)
    with LOCK:
        cam, _ = gradcam(model, x)
    grey = np.asarray(img.convert("L").resize((448, 448)), dtype=np.float32) / 255
    cam_big = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((448, 448)), dtype=np.float32) / 255
    # A one-colour glow instead of a rainbow map: brighter means a stronger push towards "bleed".
    # It stays readable for colour-blind viewers and keeps red reserved for the alert.
    glow = (cam_big ** 1.5)[..., None] * CREAM * 0.9
    over = (0.55 * np.stack([grey] * 3, -1) + glow).clip(0, 1)
    return (over * 255).astype("uint8")

@st.cache_data(show_spinner=False, max_entries=64)
def check_image(data: bytes):
    """Return None if the bytes are a readable image, otherwise (plain-English reason, the exception)."""
    if not data:
        return "The file is empty.", None
    head = data[:200].lstrip().lower()
    if head.startswith(b"version https://git-lfs"):
        return ("This file is a Git LFS pointer, not a real image. Upload the example to GitHub again as a normal file "
                "(not through Git LFS)."), None
    if head.startswith((b"<!doctype", b"<html")):
        return "This file is a saved web page with an image name, not a real image. Download the actual image file.", None
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception as e:
        return "The app could not read that file as an image. Please try a PNG or JPG.", repr(e)
    return None

# ------------------------------------------------------------------ the page
st.markdown(f"<style>{CSS_CLEAN}</style>", unsafe_allow_html=True)
st.markdown(f"<style>{FONT_CSS}</style>", unsafe_allow_html=True)   # separate block: fonts never delay the page styles
st.markdown(TOPBAR_HTML, unsafe_allow_html=True)
st.markdown(_clean(INTRO_HTML), unsafe_allow_html=True)

examples = {}
if EXAMPLES.is_dir():
    for f in sorted(EXAMPLES.iterdir()):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"} and f.name != "brain.png":
            truth = "bleed" if "bleed" in f.name.lower() else "normal"
            examples[f"Public example {len(examples) + 1} (labelled {truth} in the data set)"] = f

# Left: choose a scan. Middle: the brain. Right: the result. On a phone they stack in that order.
left, mid, right = st.columns([1, 1.15, 1], gap="large")

with left:
    st.markdown('<h2 class="bc-h2">1. Choose a scan</h2>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload a head CT slice", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    st.markdown('<p class="hint">Use one axial head CT slice as a PNG or JPG. Other kinds of images give meaningless scores.</p>', unsafe_allow_html=True)
    choice = st.selectbox("Or try a public example", ["None"] + list(examples)) if examples else "None"
    with st.expander("Settings"):
        thr = st.slider("Alert line", 0.05, 0.95, 0.50, 0.05,
                        help="The tool flags a scan when its bleeding probability is at or above this value. A lower line flags more scans. 0.50 is the setting we tested.")

data, source = None, ""
if upload is not None:
    data, source = upload.getvalue(), "Your upload"
elif choice != "None":
    data, source = examples[choice].read_bytes(), choice

# Two separate steps, so the message always matches the real problem:
#   1. is the file a readable image?   2. did the model run?
result, problem, problem_detail = None, None, None
if data is not None:
    bad = check_image(data)
    if bad is not None:
        problem, problem_detail = bad
    else:
        try:
            result = predict(data)
        except Exception as e:
            problem, problem_detail = "The image is fine, but the model could not run. The model weights may be missing or failed to download.", e

flagged = result is not None and result[0][1] >= thr

with mid:
    st.markdown(stage_html(flagged), unsafe_allow_html=True)

with right:
    st.markdown('<h2 class="bc-h2">2. Read the result</h2>', unsafe_allow_html=True)
    if problem:
        st.error(problem)
        if problem_detail is not None:
            with st.expander("Technical details"):
                if isinstance(problem_detail, BaseException):
                    st.exception(problem_detail)
                else:
                    st.code(str(problem_detail))
    elif result is None:
        st.markdown(_clean(EMPTY_HTML), unsafe_allow_html=True)
    else:
        st.markdown(verdict_html(result[0][1], thr), unsafe_allow_html=True)
        st.markdown(findings_html(result[0], thr), unsafe_allow_html=True)

# The scan and heatmap section is filled in at the very end, so the verdict and the rest of the page show first.
heat_area = st.container()

st.markdown('<div class="section"><h2 class="bc-h2">How it works</h2></div>', unsafe_allow_html=True)
st.markdown(_clean(HOW_HTML), unsafe_allow_html=True)

st.markdown('<div class="section"><h2 class="bc-h2">How it did in our tests</h2></div>', unsafe_allow_html=True)
st.markdown(results_html(RESULTS_ROWS), unsafe_allow_html=True)

st.markdown('<div class="section"><h2 class="bc-h2">How many local scans does a new hospital need?</h2></div>', unsafe_allow_html=True)
st.markdown(_clean(TUNING_INTRO_HTML), unsafe_allow_html=True)
st.markdown(tuning_html(TUNING_ROWS), unsafe_allow_html=True)
st.markdown(_clean(TUNING_NOTE_HTML), unsafe_allow_html=True)

st.markdown(_clean(LIMITS_HTML), unsafe_allow_html=True)

st.markdown('<div class="section"><h2 class="bc-h2">What we would do next</h2></div>', unsafe_allow_html=True)
st.markdown(_clean(NEXT_HTML), unsafe_allow_html=True)

st.markdown(_clean(FOOTER_HTML), unsafe_allow_html=True)

# ------------------------------------------------------------------ last: heatmap (slow) and model warm-up
if result is not None:
    probs, scan = result
    with heat_area:
        st.markdown('<div class="section"><h2 class="bc-h2">Scan and heatmap</h2></div>', unsafe_allow_html=True)
        a, b = st.columns(2, gap="large")
        with a:
            st.image(scan, output_format="JPEG", caption=f"{source}: the scan as the model received it.")
        with b:
            slot = st.empty()
            slot.markdown(LOADING_HTML, unsafe_allow_html=True)
            try:
                slot.image(heatmap(data), output_format="JPEG", caption="Heatmap (Grad-CAM). Brighter areas pushed the score towards bleed.")
            except Exception:
                slot.warning("The heatmap could not be drawn. The result above is still valid.")
        st.markdown('<p class="caption">The heatmap shows what drove the score. It does not show where a bleed is.</p>', unsafe_allow_html=True)
elif problem is None and not st.session_state.get("_warm_failed"):
    # No scan chosen yet: the page is already on screen, so load the model now while the visitor decides.
    try:
        load_model()
    except Exception:
        st.session_state["_warm_failed"] = True   # do not retry every rerun; a real scan will show the full error
