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
WEIGHTS_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{WEIGHTS_VERSION}/bleedcheck_weights.pt"
REPO_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"

# Copy these from the notebook's "Summary of results" section after the final run
RESULTS_ROWS = [
    ("Baseline: always say \"no bleed\"", "0%", "0%"),
    ("Same source as the training data", "91%", "0.7%"),
    ("A different public source", "91%", "16%"),
    ("Different source, after tuning on 100 local scans", "95%", "1%"),
]

HERE = Path(__file__).parent
WEIGHTS = HERE / f"bleedcheck_weights_{WEIGHTS_VERSION}.pt"
EXAMPLES = HERE
BRAIN_IMAGE = HERE / "assets" / "brain.png"
CLASSES = ["Normal", "Bleeding", "Ischemia"]
SHOW_ORDER = [1, 2, 0]                    # show Bleeding first, then Ischemia, then Normal

st.set_page_config(page_title="BleedCheck CT", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ look and feel
# Cream paper, steel blue, neon pink and cyan accents, and red.
# Red means one thing: "possible bleed".
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
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
.gantry::before { content: ""; position: absolute; inset: 14px; border-radius: 50%; border: 2px dashed var(--steel); }
.gantry::after { content: ""; position: absolute; inset: -20px; border-radius: 50%;
  background: repeating-conic-gradient(var(--steel) 0deg .7deg, transparent .7deg 7.5deg);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
          mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px)); }
.brain-art { width: 82%; aspect-ratio: 640 / 531; background-repeat: no-repeat; background-position: center; background-size: contain; position: relative; z-index: 1; }
.gantry.flag { border-color: var(--alert); }
.gantry.flag::before { border-color: var(--alert); }
.gantry.flag::after { background: repeating-conic-gradient(var(--alert) 0deg .7deg, transparent .7deg 7.5deg); }
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
.results { width: 100%; border-collapse: collapse; border: 2px solid var(--ink); font-size: 1rem; }
.results caption { text-align: left; padding: 0 0 8px; font-size: .95rem; caption-side: bottom; }
.results th { text-align: left; font-weight: 700; padding: 10px 14px; background: var(--ink); color: var(--paper); }
.results td { padding: 11px 14px; border-bottom: 1px solid var(--steel); }
.results tr:last-child td { border-bottom: none; }
.results td:not(:first-child), .results th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; width: 22%; }
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
  .results { font-size: .9rem; } .results th, .results td { padding: 9px 8px; }
}
@media (prefers-reduced-motion: reduce) { .stApp * { animation: none !important; transition: none !important; scroll-behavior: auto !important; } }

/* ================= NEON LAYER ================= */
:root { --pink:#FF2E9A; --cyan:#19E3FF; --violet:#8A5CFF; }

/* --- Neon ring and aura around the brain (turns red when a bleed is flagged) --- */
.gantry:not(.flag) {
  border-color: var(--pink);
  background: radial-gradient(circle, rgba(255,46,154,.16) 0%, rgba(25,227,255,.10) 55%, transparent 72%);
  animation: ringPulse 5s ease-in-out infinite;
}
.gantry:not(.flag)::before { border-color: var(--cyan); animation: spinRev 50s linear infinite; }
.gantry:not(.flag)::after {
  background: repeating-conic-gradient(var(--pink) 0deg .7deg, transparent .7deg 3.75deg, var(--cyan) 3.75deg 4.45deg, transparent 4.45deg 7.5deg);
  filter: drop-shadow(0 0 4px var(--pink)) drop-shadow(0 0 8px var(--cyan));
  animation: spin 40s linear infinite;
}
.gantry.flag { animation: alertPulse 1.6s ease-in-out infinite; }
.brain-art { animation: brainGlow 5s ease-in-out infinite; }

@keyframes spin    { to { transform: rotate(360deg); } }
@keyframes spinRev { to { transform: rotate(-360deg); } }
@keyframes ringPulse {
  0%,100% { box-shadow: 0 0 10px 1px var(--pink), 0 0 26px 4px rgba(255,46,154,.45), inset 0 0 18px rgba(255,46,154,.35); border-color: var(--pink); }
  50%     { box-shadow: 0 0 10px 1px var(--cyan), 0 0 30px 6px rgba(25,227,255,.45), inset 0 0 22px rgba(25,227,255,.35); border-color: var(--cyan); }
}
@keyframes alertPulse {
  0%,100% { box-shadow: 0 0 10px 1px var(--alert), 0 0 24px 4px rgba(204,0,0,.4); }
  50%     { box-shadow: 0 0 16px 3px var(--alert), 0 0 40px 10px rgba(204,0,0,.5); }
}
@keyframes brainGlow {
  0%,100% { filter: drop-shadow(0 0 6px var(--pink)) drop-shadow(0 0 18px rgba(255,46,154,.6)); }
  50%     { filter: drop-shadow(0 0 6px var(--cyan)) drop-shadow(0 0 18px rgba(138,92,255,.6)); }
}

/* --- A light that travels down the page: top bar, then each section lights up in turn --- */
.topbar { border-bottom: 3px solid transparent; border-image: linear-gradient(90deg, var(--steel), var(--pink), var(--cyan), var(--violet), var(--steel)) 1; border-image-slice: 1; }
.topbar { background: linear-gradient(90deg, transparent, rgba(255,46,154,.10), rgba(25,227,255,.10), transparent) 0 100% / 300% 3px no-repeat; animation: sweep 6s linear infinite; }
@keyframes sweep { from { background-position: 200% 100%; } to { background-position: -100% 100%; } }

.hero, .notice, .findings, .verdict, .results, .limits, .empty { transition: box-shadow .25s ease, transform .25s ease, border-color .25s ease; }
.hero    { animation: hop 18s ease-in-out infinite 0s; }
.notice  { animation: hop 18s ease-in-out infinite 3s; }
.verdict, .empty { animation: hop 18s ease-in-out infinite 6s; }
.findings { animation: hop 18s ease-in-out infinite 9s; }
.results { animation: hop 18s ease-in-out infinite 12s; }
.limits  { animation: hop 18s ease-in-out infinite 15s; }
@keyframes hop {
  0%   { box-shadow: 0 0 0 0 rgba(255,46,154,0); }
  4%, 14% { box-shadow: 0 0 12px 2px rgba(255,46,154,.75), 0 0 30px 6px rgba(25,227,255,.45); }
  18%, 100% { box-shadow: 0 0 0 0 rgba(255,46,154,0); }
}

/* --- Hover effects --- */
.hero:hover, .notice:hover, .findings:hover, .empty:hover, .limits:hover {
  transform: translateY(-3px);
  box-shadow: 0 0 14px 2px var(--pink), 0 0 34px 8px rgba(25,227,255,.45);
  animation-play-state: paused;
}
.verdict:not(.flag):hover { transform: translateY(-3px); box-shadow: 0 0 14px 2px var(--cyan), 0 0 34px 8px rgba(138,92,255,.4); }
.verdict.flag:hover { transform: translateY(-3px); box-shadow: 0 0 16px 3px var(--alert), 0 0 38px 10px rgba(204,0,0,.45); }
.results tr:hover td { background: rgba(255,46,154,.10); box-shadow: inset 0 0 10px rgba(25,227,255,.5); }
.bc-row:hover .bc-track { box-shadow: 0 0 10px var(--cyan), 0 0 22px rgba(255,46,154,.5); }
.bc-brand { transition: text-shadow .25s ease; }
.bc-brand:hover { text-shadow: 0 0 8px var(--pink), 0 0 18px var(--cyan); }
.stApp .bc-h2 { transition: text-shadow .25s ease; }
.stApp .bc-h2:hover { text-shadow: 0 0 6px rgba(255,46,154,.7), 0 0 16px rgba(25,227,255,.6); }
[data-testid="stImage"] img { transition: box-shadow .25s ease, transform .25s ease; }
[data-testid="stImage"] img:hover { transform: scale(1.015); box-shadow: 0 0 14px 2px var(--pink), 0 0 30px 6px rgba(25,227,255,.5); }
.stButton > button, [data-testid="stFileUploaderDropzone"] button { transition: box-shadow .2s ease, transform .2s ease, background .2s ease; }
.stButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover { transform: translateY(-2px); box-shadow: 0 0 12px var(--pink), 0 0 26px rgba(25,227,255,.6); }
[data-testid="stFileUploaderDropzone"], [data-testid="stExpander"], [data-baseweb="select"] > div { transition: box-shadow .25s ease, border-color .25s ease; }
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stExpander"]:hover, [data-baseweb="select"] > div:hover {
  border-color: var(--pink); box-shadow: 0 0 12px var(--pink), 0 0 26px rgba(25,227,255,.45);
}

/* Motion off for anyone who asks for it (the pseudo-elements need their own rule) */
@media (prefers-reduced-motion: reduce) {
  .stApp *, .gantry::before, .gantry::after { animation: none !important; transition: none !important; }
}
"""

def _clean(html):
    # Streamlit's markdown breaks on blank or indented lines inside HTML, so squash them
    return " ".join(line.strip() for line in html.splitlines() if line.strip())

CHECK_SVG = ('<svg viewBox="0 0 32 32" aria-hidden="true" focusable="false"><circle cx="16" cy="16" r="14" fill="none" stroke="#1F3F57" stroke-width="2.5"/>'
             '<path d="M9 16.5l5 5 9-11" fill="none" stroke="#1F3F57" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')
FLAG_SVG = ('<svg viewBox="0 0 32 32" aria-hidden="true" focusable="false"><path d="M16 3L30 28H2z" fill="#CC0000"/>'
            '<path d="M16 12v8" stroke="#FFFFF0" stroke-width="3" stroke-linecap="round"/><circle cx="16" cy="24" r="1.8" fill="#FFFFF0"/></svg>')

TOPBAR_HTML = '<div class="topbar"><span class="bc-brand">BleedCheck CT</span><span class="tag">A student prototype for the Veritas AI Hackathon 2026</span></div>'

INTRO_HTML = """
<div class="hero">
<h1 class="bc-title">A second look at head CT scans</h1>
<p class="lede">Upload one head CT slice. BleedCheck estimates the chance of a brain bleed and shows what drove that estimate, so an emergency doctor can decide which scans to open first.</p>
<p>We built it for the doctor on a night shift with no radiologist in the building. In our tests it caught 91 of every 100 bleeds, and it missed about 1 in 11.</p>
</div>
<div class="notice">
<p><b>This project is a prototype and is not medical advice.</b></p>
<p>Please do not upload a real patient's scan. The app processes images in memory and does not store them.</p>
</div>
"""

EMPTY_HTML = """
<div class="empty"><b>No scan selected yet.</b><br>Upload an image on the left, or choose a public example. The result and the heatmap appear here.</div>
"""

BRAIN_ALT = "Side view of a human brain, shaded in steel blue and cream. This is an illustration, not your scan."
BRAIN_PATHS = [HERE / "assets" / "brain.png", HERE / "brain.png", HERE.parent / "assets" / "brain.png",
               HERE.parent / "app" / "assets" / "brain.png"]

@st.cache_data
def _read_b64(path_str):
    return base64.b64encode(Path(path_str).read_bytes()).decode()

def brain_b64():
    for p in BRAIN_PATHS:
        if p.exists():
            return _read_b64(str(p))
    return ""   # not cached, so adding the file later works without a reboot

def stage_html(flagged):
    b64 = brain_b64()
    # The image goes in CSS (a data URI in an <img> tag can be stripped by Streamlit's markdown renderer).
    css = f'<style>.brain-art {{ background-image: url("data:image/png;base64,{b64}"); }}</style>' if b64 else ""
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

def results_html(rows):
    body = ""
    for n, (a, b, c) in enumerate(rows):
        tr = '<tr class="base">' if n == 0 else "<tr>"
        body += f"{tr}<td>{a}</td><td>{b}</td><td>{c}</td></tr>"
    return _clean(f'<table class="results"><caption>A false alarm is a normal scan that the model flagged as a possible bleed.</caption>'
                  f'<tr><th scope="col">Test</th><th scope="col">Bleeds caught</th><th scope="col">False alarms</th></tr>{body}</table>')

LIMITS_HTML = """
<aside class="limits"><h3>Read the limits before you trust a result</h3>
<p>On scans like its training data, the model missed about 1 in 11 bleeds.</p>
<p>On scans from a different public source, it wrongly flagged 10 to 20% of normal scans. A few labelled scans from the new source brought that down a lot.</p>
<p>The heatmap shows what drove the score, not where a bleed is. It often lights up the skull edge or the scan background.</p>
<p>Our data sets are small and public: they have no patient IDs, and the extra source holds only 200 scans. Treat the numbers as a study, not a guarantee.</p></aside>
"""

FOOTER_HTML = f"""
<div class="footer"><b>This project is a prototype and is not medical advice.</b><br>
Made by Aarish Asif Khan and Aabir Asif Khan, Islamabad. Code and notebook: <a href="{REPO_URL}">GitHub</a>.<br>
We trained on the Brain Stroke CT Dataset, a copy of the TEKNOFEST-2021 stroke data set from the Republic of Türkiye Ministry of Health (Koç et al., Eurasian J Med 2022).
We tested on Head CT - hemorrhage by Felipe Kitamura (CC0). The example scans come from the CC0 set. We built the app with PyTorch, torchvision and Streamlit.</div>
"""

# ------------------------------------------------------------------ the model
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
LOCK = threading.Lock()      # the cached model is shared, so run one analysis at a time
CREAM = np.array([1.0, 1.0, 0.94], dtype=np.float32)

@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def load_model():
    if not WEIGHTS.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
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
def analyse(data: bytes):
    model = load_model()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = TF(img).unsqueeze(0)
    with LOCK:
        cam, p = gradcam(model, x)
    grey = np.asarray(img.convert("L").resize((448, 448)), dtype=np.float32) / 255
    cam_big = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((448, 448)), dtype=np.float32) / 255
    # A one-colour glow instead of a rainbow map: brighter means a stronger push towards "bleed".
    # It stays readable for colour-blind viewers and keeps red reserved for the alert.
    glow = (cam_big ** 1.5)[..., None] * CREAM * 0.9
    over = (0.55 * np.stack([grey] * 3, -1) + glow).clip(0, 1)
    return [float(v) for v in p], (grey * 255).astype("uint8"), (over * 255).astype("uint8")

# ------------------------------------------------------------------ the page
st.markdown(f"<style>{_clean(CSS)}</style>", unsafe_allow_html=True)
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

result, failed = None, False
if data is not None:
    try:
        result = analyse(data)
    except Exception:
        failed = True

flagged = result is not None and result[0][1] >= thr

with mid:
    st.markdown(stage_html(flagged), unsafe_allow_html=True)

with right:
    st.markdown('<h2 class="bc-h2">2. Read the result</h2>', unsafe_allow_html=True)
    if failed:
        st.error("The app could not read that file as an image. Please try a PNG or JPG.")
    elif result is None:
        st.markdown(_clean(EMPTY_HTML), unsafe_allow_html=True)
    else:
        st.markdown(verdict_html(result[0][1], thr), unsafe_allow_html=True)
        st.markdown(findings_html(result[0], thr), unsafe_allow_html=True)

if result is not None:
    probs, scan, over = result
    st.markdown('<div class="section"><h2 class="bc-h2">Scan and heatmap</h2></div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.image(scan, caption=f"{source}: the scan as the model received it.")
    with b:
        st.image(over, caption="Heatmap (Grad-CAM). Brighter areas pushed the score towards bleed.")
    st.markdown('<p class="caption">The heatmap shows what drove the score. It does not show where a bleed is.</p>', unsafe_allow_html=True)

st.markdown('<div class="section"><h2 class="bc-h2">How it did in our tests</h2></div>', unsafe_allow_html=True)
st.markdown(results_html(RESULTS_ROWS), unsafe_allow_html=True)
st.markdown(_clean(LIMITS_HTML), unsafe_allow_html=True)
st.markdown(_clean(FOOTER_HTML), unsafe_allow_html=True)
