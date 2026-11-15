"""BleedCheck CT: a research prototype. Not medical advice."""
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
GITHUB_REPO = "veritas-hacakthon-2026"            # <- put the exact repository name here
WEIGHTS_VERSION = "v1"                    # the release tag holding the weights; use "v2" if you publish new weights later
WEIGHTS_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{WEIGHTS_VERSION}/bleedcheck_weights.pt"
REPO_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"

# Copy these from the notebook's "Summary of results" section after the final run
RESULTS_ROWS = [
    ("Same source as the training data", "91%", "0.7%"),
    ("A different public source", "91%", "16%"),
    ("Different source, after tuning on 100 local scans", "95%", "1%"),
]

HERE = Path(__file__).parent
WEIGHTS = HERE / f"bleedcheck_weights_{WEIGHTS_VERSION}.pt"
EXAMPLES = HERE / "examples"
CLASSES = ["Normal", "Bleeding", "Ischemia"]

st.set_page_config(page_title="BleedCheck CT", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ look and feel (plain HTML + CSS, no scripts)
CSS = """
html, body, [data-testid="stAppViewContainer"], .stApp { background: #F6F6F9; }
html, body, .stApp, .stApp * { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1080px; padding: 1.6rem 1.5rem 3rem; }
.topbar { display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 14px; border-bottom: 1px solid #E3E5EC; margin-bottom: 28px; }
.brand { font-weight: 700; font-size: 1.05rem; color: #1F2430; }
.tag { font-size: .85rem; color: #6A7083; }
.page-title { font-size: 2rem; line-height: 1.2; font-weight: 650; letter-spacing: -.015em; color: #1F2430; margin: 0 0 10px; }
.lede { font-size: 1.05rem; line-height: 1.55; color: #454B5E; max-width: 680px; margin: 0 0 18px; }
.notice { border-left: 3px solid #C77700; background: #FFF8EB; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: .92rem; color: #5C3D00; line-height: 1.5; margin-bottom: 26px; }
.step { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #6A7083; font-weight: 600; margin: 4px 0 8px; }
.hint { font-size: .88rem; color: #6A7083; line-height: 1.5; margin: 6px 0 12px; }
.empty { border: 1px dashed #C9CDDA; border-radius: 12px; padding: 28px 22px; color: #6A7083; background: #FBFBFD; line-height: 1.55; }
.empty b { color: #1F2430; }
.verdict { background: #FFFFFF; border: 1px solid #E3E5EC; border-left: 4px solid #4B4FC4; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; }
.verdict.flag { border-left-color: #C2255C; }
.v-title { font-size: 1.25rem; font-weight: 650; color: #1F2430; margin-bottom: 4px; }
.verdict p { margin: 0 0 14px; color: #454B5E; font-size: .95rem; line-height: 1.5; }
.meter { position: relative; height: 8px; background: #E7E9F1; border-radius: 4px; }
.meter-fill { height: 100%; background: #4B4FC4; border-radius: 4px; }
.verdict.flag .meter-fill { background: #C2255C; }
.meter-tick { position: absolute; top: -5px; width: 2px; height: 18px; background: #1F2430; }
.meter-labels { display: flex; justify-content: space-between; font-size: .82rem; color: #6A7083; margin-top: 8px; }
.classes { background: #FFFFFF; border: 1px solid #E3E5EC; border-radius: 10px; padding: 6px 16px; margin-bottom: 14px; }
.cls-row { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid #EEF0F5; font-size: .95rem; color: #1F2430; }
.cls-row:last-child { border-bottom: none; }
.cls-row span:last-child { font-variant-numeric: tabular-nums; color: #454B5E; }
.caption { font-size: .85rem; color: #6A7083; line-height: 1.45; }
.section-title { font-size: 1.25rem; font-weight: 650; color: #1F2430; margin: 34px 0 6px; }
.results { width: 100%; border-collapse: collapse; background: #FFFFFF; border: 1px solid #E3E5EC; border-radius: 10px; overflow: hidden; font-size: .95rem; }
.results th { text-align: left; font-weight: 600; font-size: .82rem; color: #6A7083; padding: 10px 14px; background: #FBFBFD; border-bottom: 1px solid #E3E5EC; }
.results td { padding: 11px 14px; border-bottom: 1px solid #EEF0F5; color: #1F2430; }
.results tr:last-child td { border-bottom: none; }
.results td:not(:first-child), .results th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; width: 24%; }
.small { font-size: .88rem; color: #6A7083; line-height: 1.55; }
.small a, .footer a { color: #4B4FC4; }
.footer { margin-top: 38px; padding-top: 16px; border-top: 1px solid #E3E5EC; font-size: .85rem; color: #6A7083; line-height: 1.6; }
@media (max-width: 1024px) { .block-container, [data-testid="stMainBlockContainer"] { padding: 1.3rem 1.2rem 2.5rem; } .page-title { font-size: 1.75rem; } }
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1rem .9rem 2rem; }
  .topbar { flex-direction: column; gap: 2px; margin-bottom: 20px; }
  .page-title { font-size: 1.45rem; } .lede { font-size: 1rem; }
  .verdict { padding: 14px; } .v-title { font-size: 1.15rem; }
  .results { font-size: .85rem; } .results th, .results td { padding: 9px 8px; } .results td:not(:first-child), .results th:not(:first-child) { width: 25%; }
}
"""

def _clean(html):
    # Streamlit's markdown breaks on blank or indented lines inside HTML, so squash them
    return " ".join(line.strip() for line in html.splitlines() if line.strip())

TOPBAR_HTML = '<div class="topbar"><span class="brand">BleedCheck CT</span><span class="tag">Research prototype, built for the Veritas AI Hackathon 2026</span></div>'

INTRO_HTML = """
<h1 class="page-title">Check a head CT slice for a possible brain bleed</h1>
<p class="lede">Upload one axial head CT image, or pick a public example. The model gives a bleed score and a heatmap of what pushed that score.
It is a student project to study how such models behave on new data, not a diagnostic tool.</p>
<div class="notice"><b>This project is a prototype and is not medical advice.</b> Please do not upload a real patient's scan. Images are processed in memory and are not stored.</div>
"""

EMPTY_HTML = """
<div class="empty"><b>No scan selected yet.</b><br>Upload an image on the left, or choose one of the public examples.
The result and the heatmap will appear here.</div>
"""

def verdict_html(p, thr):
    flagged = p >= thr
    cls = "flag" if flagged else "clear"
    title = "Possible bleed flagged" if flagged else "No bleed flagged"
    text = ("The model's bleed score is above the threshold. Treat this as a reason for a closer look, not as a diagnosis."
            if flagged else "The model's bleed score is below the threshold. This does not rule out a bleed.")
    return _clean(f"""
    <div class="verdict {cls}"><div class="v-title">{title}</div><p>{text}</p>
    <div class="meter"><div class="meter-fill" style="width:{min(max(p, 0), 1) * 100:.1f}%"></div><div class="meter-tick" style="left:{thr * 100:.1f}%"></div></div>
    <div class="meter-labels"><span>Bleed score {p:.2f}</span><span>Threshold {thr:.2f}</span></div></div>""")

def classes_html(probs):
    rows = "".join(f'<div class="cls-row"><span>{n}</span><span>{v * 100:.0f}%</span></div>' for n, v in zip(CLASSES, probs))
    return _clean(f'<div class="classes">{rows}</div>')

def results_html(rows):
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return _clean(f'<table class="results"><tr><th>Test</th><th>Bleeds caught</th><th>False alarms</th></tr>{body}</table>')

LIMITS_HTML = """
<p class="small"><b>What to keep in mind.</b>
In our tests the model missed roughly 1 in 11 bleeds, even on data like its training data.
On scans from a different public source it wrongly flagged 10 to 20% of normal scans, and a few labelled scans from the new source brought that down a lot.
The heatmap shows what drove the score, not where a bleed is: it often highlights the skull edge or the scan background.
The figures come from small public data sets (no patient IDs, one extra source of 200 scans), so treat them as a study, not a guarantee.</p>
"""

FOOTER_HTML = f"""
<div class="footer"><b>This project is a prototype and is not medical advice.</b><br>
Made by Aarish Asif Khan and Aabir Asif Khan, Islamabad. Code and notebook: <a href="{REPO_URL}">GitHub</a>.<br>
Trained on the Brain Stroke CT Dataset (a copy of the TEKNOFEST-2021 stroke data set, Republic of Türkiye Ministry of Health; Koç et al., Eurasian J Med 2022).
Tested on Head CT - hemorrhage by Felipe Kitamura (CC0). The example scans come from the CC0 set. Built with PyTorch, torchvision and Streamlit.</div>
"""

# ------------------------------------------------------------------ the model
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
LOCK = threading.Lock()      # the cached model is shared, so run one analysis at a time

@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def load_model():
    if not WEIGHTS.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval()

def jet(x):
    # a small stand-in for matplotlib's "jet" colour map
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)

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
    over = (0.6 * np.stack([grey] * 3, -1) + 0.4 * jet(cam_big)).clip(0, 1)
    return [float(v) for v in p], (grey * 255).astype("uint8"), (over * 255).astype("uint8")

# ------------------------------------------------------------------ the page
st.markdown(f"<style>{_clean(CSS)}</style>", unsafe_allow_html=True)
st.markdown(TOPBAR_HTML, unsafe_allow_html=True)
st.markdown(_clean(INTRO_HTML), unsafe_allow_html=True)

examples = {}
if EXAMPLES.is_dir():
    for f in sorted(EXAMPLES.iterdir()):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            truth = "bleed" if "bleed" in f.name.lower() else "normal"
            examples[f"Public example {len(examples) + 1} (labelled {truth} in the data set)"] = f

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="step">1. Choose a scan</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload a head CT slice", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    st.markdown('<p class="hint">One axial head CT slice as a PNG or JPG. Other kinds of images give meaningless scores.</p>', unsafe_allow_html=True)
    choice = st.selectbox("Or try a public example", ["None"] + list(examples)) if examples else "None"
    with st.expander("Settings"):
        thr = st.slider("Flag threshold", 0.05, 0.95, 0.50, 0.05,
                        help="A scan is flagged when its bleed score is at or above this value. Lower flags more scans. 0.50 is the setting we tested.")

data, source = None, ""
if upload is not None:
    data, source = upload.getvalue(), "Your upload"
elif choice != "None":
    data, source = examples[choice].read_bytes(), choice

result = None
with right:
    st.markdown('<div class="step">2. Result</div>', unsafe_allow_html=True)
    if data is None:
        st.markdown(_clean(EMPTY_HTML), unsafe_allow_html=True)
    else:
        try:
            probs, scan, over = analyse(data)
        except Exception:
            st.error("That file could not be read as an image. Please try a PNG or JPG.")
        else:
            result = (probs, scan, over)
            st.markdown(verdict_html(probs[1], thr), unsafe_allow_html=True)
            st.markdown(classes_html(probs), unsafe_allow_html=True)

if result is not None:
    st.markdown('<div class="section-title">Scan and heatmap</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.image(scan, caption=source)
    with b:
        st.image(over, caption="Where the model looked (Grad-CAM). Red and yellow pushed the score towards bleed.")
    st.markdown('<p class="caption">The heatmap shows what drove the score. It does not show where a bleed is.</p>', unsafe_allow_html=True)

st.markdown('<div class="section-title">How it did in our tests</div>', unsafe_allow_html=True)
st.markdown(results_html(RESULTS_ROWS), unsafe_allow_html=True)
st.markdown('<p class="caption" style="margin-top:8px">A false alarm is a normal scan that the model flagged as a possible bleed.</p>', unsafe_allow_html=True)
st.markdown(_clean(LIMITS_HTML), unsafe_allow_html=True)
st.markdown(_clean(FOOTER_HTML), unsafe_allow_html=True)
