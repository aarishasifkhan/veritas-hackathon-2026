"""BleedCheck CT: a research prototype. Not medical advice."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms as T
import matplotlib.cm as cm
import gradio as gr

CLASSES = ["Normal", "Bleeding", "Ischemia"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
WEIGHTS = next((p for p in ["E0_plain_s3.pt", "/kaggle/working/E0_plain_s3.pt"] if os.path.exists(p)), "E0_plain_s3.pt")
device = "cpu"  # small model, CPU is enough for one image

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.to(device).eval()

tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def gradcam(x, class_idx=1):
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

# ---- palette: change these colours and nothing else needs touching ----
PALETTE = {
    "bg": "#F3F4FB",          # page background
    "card": "#FFFFFF",        # panels
    "text": "#23264A",        # main text
    "muted": "#585D78",       # small text
    "accent": "#5B5BD6",      # buttons, highlights
    "accent_dark": "#3F3FB0", # headings, big numbers
    "soft": "#ECEEFC",        # light tint of the accent
    "line": "#E0E3F3",        # borders
    "flag": "#D6336C",        # "possible bleed" colour
    "flag_bg": "#FFF0F5",     # "possible bleed" background
    "warn": "#9A4A06",        # disclaimer text
    "warn_bg": "#FFF6E5",     # disclaimer background
}

CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
.gradio-container, .dark .gradio-container {
  --body-background-fill: __BG__; --background-fill-primary: __CARD__; --background-fill-secondary: __BG__;
  --block-background-fill: __CARD__; --block-border-color: __LINE__; --border-color-primary: __LINE__;
  --body-text-color: __TEXT__; --block-label-text-color: __MUTED__; --block-title-text-color: __TEXT__;
  --input-background-fill: __CARD__; --button-primary-background-fill: __ACCENT__;
  --button-primary-background-fill-hover: __ACCENT_DARK__; --button-primary-text-color: #FFFFFF;
  --color-accent: __ACCENT__; --slider-color: __ACCENT__;
  --block-radius: 20px; --button-large-radius: 14px; --block-shadow: 0 18px 36px -24px rgba(63,63,176,.45);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  background: radial-gradient(900px 500px at 0% 0%, #E6E8FB, transparent), radial-gradient(800px 500px at 100% 30%, #F8E3EE, transparent), __BG__ !important;
  color: __TEXT__ !important; max-width: 1120px !important; margin: 0 auto;
}
footer { display: none !important; }
.bc-bare { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }

/* ---------- hero: giant word behind a floating CT slice ---------- */
.bc-hero { position: relative; overflow: hidden; height: 470px; border-radius: 30px; perspective: 900px;
  background: radial-gradient(700px 320px at 15% 0%, #FFFFFF, transparent), radial-gradient(600px 360px at 95% 100%, #F5D3E6, transparent),
              linear-gradient(180deg, #F4F5FD, #DDE0F6); box-shadow: 0 30px 60px -34px rgba(63,63,176,.55); }
.bc-bar { position: absolute; z-index: 5; top: 20px; left: 24px; right: 24px; display: flex; justify-content: space-between; align-items: center; }
.bc-brand { font-family: 'Sora', sans-serif; font-weight: 700; color: __ACCENT_DARK__; font-size: 1rem; letter-spacing: .02em; }
.bc-pill { padding: 5px 14px; border-radius: 999px; font-size: .78rem; font-weight: 600; color: __ACCENT_DARK__;
  background: rgba(255,255,255,.7); border: 1px solid rgba(255,255,255,.95); backdrop-filter: blur(10px); }
.bc-giant { position: absolute; z-index: 1; left: 0; right: 0; top: 44%; transform: translateY(-50%); text-align: center;
  font-family: 'Sora', sans-serif; font-weight: 800; line-height: 1; letter-spacing: -.02em; white-space: nowrap;
  font-size: clamp(2.7rem, 11vw, 8.8rem); background: linear-gradient(180deg, #8A8FE0 0%, #C9CCF4 85%);
  -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; user-select: none; }
.bc-stage { position: absolute; z-index: 3; left: 50%; top: 44%; width: 224px; height: 262px; margin: -131px 0 0 -112px; animation: bc-float 6s ease-in-out infinite; }
.bc-disc { position: relative; width: 100%; height: 100%; border-radius: 50%; overflow: hidden;
  box-shadow: 0 0 0 6px rgba(255,255,255,.55), 0 30px 60px -10px rgba(35,38,74,.65), 0 0 90px 10px rgba(91,91,214,.45); }
.bc-disc svg { width: 100%; height: 100%; display: block; }
.bc-scan { position: absolute; left: 0; right: 0; height: 46px; top: -20%;
  background: linear-gradient(180deg, transparent, rgba(190,196,255,.75), transparent); mix-blend-mode: screen; animation: bc-scan 3.6s ease-in-out infinite; }
.bc-hot { animation: bc-pulse 2.4s ease-in-out infinite; transform-origin: 262px 250px; }
.bc-pod { position: absolute; z-index: 2; left: 50%; bottom: 40px; width: 360px; height: 62px; margin-left: -180px; border-radius: 50%;
  background: radial-gradient(ellipse at center, rgba(255,255,255,.95) 0%, rgba(160,165,240,.55) 45%, rgba(91,91,214,0) 72%);
  border: 2px solid rgba(255,255,255,.9); box-shadow: 0 0 40px 6px rgba(91,91,214,.35), inset 0 0 20px rgba(255,255,255,.9); }
.bc-sub { position: absolute; z-index: 5; left: 26px; bottom: 22px; right: 26px; display: flex; justify-content: space-between; align-items: end; gap: 14px; flex-wrap: wrap; }
.bc-sub p { margin: 0; max-width: 430px; font-size: 1.02rem; color: __TEXT__; font-weight: 500; line-height: 1.4; }
.bc-chip { padding: 7px 14px; border-radius: 12px; font-size: .8rem; font-weight: 600; color: #FFFFFF;
  background: linear-gradient(135deg, __ACCENT_DARK__, __ACCENT__); box-shadow: 0 10px 20px -8px rgba(63,63,176,.7); }
@keyframes bc-float { 0%,100% { transform: translateY(0) rotateY(-9deg) rotateX(3deg); } 50% { transform: translateY(-14px) rotateY(9deg) rotateX(-2deg); } }
@keyframes bc-scan { 0% { top: -20%; } 100% { top: 105%; } }
@keyframes bc-pulse { 0%,100% { transform: scale(1); opacity: .85; } 50% { transform: scale(1.35); opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .bc-stage, .bc-scan, .bc-hot { animation: none; } }

/* ---------- glass cards ---------- */
.bc-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0; }
.bc-glass { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.95); border-radius: 22px; backdrop-filter: blur(14px);
  box-shadow: 0 22px 40px -26px rgba(63,63,176,.55); transition: transform .25s ease, box-shadow .25s ease; }
.bc-glass:hover { transform: perspective(700px) rotateX(4deg) translateY(-5px); box-shadow: 0 30px 46px -24px rgba(63,63,176,.6); }
.bc-stat { padding: 18px 20px; }
.bc-stat b { display: block; font-family: 'Sora', sans-serif; font-size: 2rem; line-height: 1.1; color: __ACCENT_DARK__; }
.bc-stat span { font-size: .9rem; color: __MUTED__; }
.bc-warn { border: 1px solid #F0CF9B; background: __WARN_BG__; border-radius: 14px; padding: 11px 16px; color: __WARN__; font-size: .93rem; margin-bottom: 8px; }

/* ---------- result ---------- */
.bc-verdict { display: flex; gap: 18px; align-items: center; padding: 18px 20px; border-radius: 22px; border: 1px solid __LINE__; background: __CARD__;
  box-shadow: 0 22px 40px -28px rgba(63,63,176,.55); }
.bc-verdict.flag { border-color: __FLAG__; background: linear-gradient(135deg, __FLAG_BG__, #FFFFFF); }
.bc-verdict.clear { border-color: __ACCENT__; background: linear-gradient(135deg, __SOFT__, #FFFFFF); }
.bc-ring { --p: 0; --c: __ACCENT__; flex: none; width: 96px; height: 96px; border-radius: 50%; position: relative; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), #E1E4F4 0); }
.bc-ring::before { content: ""; position: absolute; inset: 11px; border-radius: 50%; background: #FFFFFF; }
.bc-ring b { position: relative; font-family: 'Sora', sans-serif; font-size: 1.25rem; color: __TEXT__; }
.bc-verdict.flag .bc-ring { --c: __FLAG__; }
.bc-verdict h3 { margin: 0 0 6px; font-family: 'Sora', sans-serif; font-size: 1.2rem; color: __TEXT__; }
.bc-verdict p { margin: 0; font-size: .93rem; color: #3A3F5C; line-height: 1.45; }
.bc-bars { margin-top: 12px; padding: 12px 18px; }
.bc-row { display: grid; grid-template-columns: 82px 1fr 52px; gap: 10px; align-items: center; margin: 9px 0; font-size: .92rem; }
.bc-track { height: 10px; background: #E1E4F4; border-radius: 999px; overflow: hidden; }
.bc-fill { height: 100%; background: #7B82A8; border-radius: 999px; }
.bc-fill.main { background: linear-gradient(90deg, __ACCENT__, #8E7CF0); }
.bc-num { text-align: right; font-variant-numeric: tabular-nums; color: __TEXT__; }

/* ---------- steps + notes ---------- */
.bc-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 18px 0 8px; }
.bc-step { padding: 18px 20px; }
.bc-step i { display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 10px; font-style: normal; font-weight: 700; color: #FFFFFF;
  background: linear-gradient(135deg, __ACCENT_DARK__, __ACCENT__); margin-bottom: 8px; }
.bc-step h4 { margin: 0 0 4px; font-family: 'Sora', sans-serif; font-size: 1.02rem; color: __TEXT__; }
.bc-step p { margin: 0; font-size: .9rem; color: __MUTED__; line-height: 1.45; }
.bc-note { font-size: .9rem; color: __MUTED__; line-height: 1.55; margin-top: 12px; }
.bc-note b { color: __TEXT__; }
@media (max-width: 760px) {
  .bc-stats, .bc-steps { grid-template-columns: 1fr; }
  .bc-hero { height: 520px; }
  .bc-giant { top: 36%; font-size: 3.2rem; }
  .bc-stage { top: 36%; width: 160px; height: 188px; margin: -94px 0 0 -80px; }
  .bc-pod { bottom: 150px; width: 230px; height: 44px; margin-left: -115px; }
  .bc-sub { flex-direction: column; align-items: flex-start; }
  .bc-sub p { font-size: .92rem; }
}
"""
CSS = CSS_TEMPLATE
for _k, _v in PALETTE.items():
    CSS = CSS.replace("__" + _k.upper() + "__", _v)

HERO_HTML = """
<div class="bc-hero">
  <div class="bc-bar"><span class="bc-brand">BleedCheck CT</span><span class="bc-pill">Research prototype</span></div>
  <div class="bc-giant">BLEEDCHECK</div>
  <div class="bc-pod"></div>
  <div class="bc-stage"><div class="bc-disc">
    <svg viewBox="0 0 400 460" xmlns="http://www.w3.org/2000/svg" aria-label="Illustration of a head CT slice">
      <defs>
        <radialGradient id="bcBone" cx="50%" cy="45%" r="60%"><stop offset="0" stop-color="#30346E"/><stop offset="1" stop-color="#0B0D2B"/></radialGradient>
        <radialGradient id="bcBrain" cx="50%" cy="42%" r="62%"><stop offset="0" stop-color="#7479C8"/><stop offset=".7" stop-color="#3B3F85"/><stop offset="1" stop-color="#1D2050"/></radialGradient>
        <linearGradient id="bcRim" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset=".5" stop-color="#C8CCFA"/><stop offset="1" stop-color="#8E92E8"/></linearGradient>
        <radialGradient id="bcHot" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#FFB4CE"/><stop offset=".55" stop-color="#F0527F" stop-opacity=".75"/><stop offset="1" stop-color="#F0527F" stop-opacity="0"/></radialGradient>
        <filter id="bcTex" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.03 0.045" numOctaves="3" seed="11" result="n"/>
          <feColorMatrix in="n" type="matrix" values="0 0 0 0 0.80  0 0 0 0 0.82  0 0 0 0 1  0 0 0 1.5 -0.55" result="m"/>
          <feComposite in="m" in2="SourceGraphic" operator="in"/>
        </filter>
      </defs>
      <rect width="400" height="460" fill="#0B0D2B"/>
      <ellipse cx="200" cy="230" rx="176" ry="208" fill="none" stroke="url(#bcRim)" stroke-width="16"/>
      <ellipse cx="200" cy="230" rx="160" ry="192" fill="url(#bcBone)"/>
      <ellipse cx="200" cy="230" rx="146" ry="176" fill="url(#bcBrain)"/>
      <ellipse cx="200" cy="230" rx="146" ry="176" fill="#000" filter="url(#bcTex)" opacity=".75"/>
      <path d="M200 70 C197 150 203 310 200 392" stroke="#171A47" stroke-width="3" fill="none" opacity=".7"/>
      <path d="M188 170 C166 184 164 258 186 284 C196 256 196 208 188 170Z" fill="#0A0C2A" opacity=".88"/>
      <path d="M212 170 C234 184 236 258 214 284 C204 256 204 208 212 170Z" fill="#0A0C2A" opacity=".88"/>
      <ellipse class="bc-hot" cx="262" cy="250" rx="34" ry="30" fill="url(#bcHot)"/>
    </svg>
    <div class="bc-scan"></div>
  </div></div>
  <div class="bc-sub">
    <p>Upload one head CT slice. See whether our model flags a possible brain bleed, and what drove the score.</p>
    <span class="bc-chip">Not medical advice</span>
  </div>
</div>"""

STATS_HTML = """<div class="bc-stats">
<div class="bc-glass bc-stat"><b>~91%</b><span>of bleeds caught on data like its training data</span></div>
<div class="bc-glass bc-stat"><b>10–20%</b><span>of normal scans wrongly flagged on a second source</span></div>
<div class="bc-glass bc-stat"><b>20% → 1%</b><span>false alarms after tuning on 100 local scans (our test)</span></div></div>"""

WARN_HTML = """<div class="bc-warn"><b>This project is a prototype and is not medical advice.</b>
Please upload only public or example images, never a real patient's scan. Your image is processed in memory and is not saved.</div>"""

EMPTY_HTML = """<div class="bc-verdict"><div class="bc-ring" style="--p:0"><b>–</b></div>
<div><h3>Waiting for a scan</h3><p>Upload a head CT slice or click an example below, then press <b>Analyse</b>.</p></div></div>"""

STEPS_HTML = """<div class="bc-steps">
<div class="bc-glass bc-step"><i>1</i><h4>Upload a slice</h4><p>One axial head CT image (PNG or JPG). Public or example images only.</p></div>
<div class="bc-glass bc-step"><i>2</i><h4>The model scores it</h4><p>A ResNet18 trained on public CT data gives Normal, Bleeding and Ischemia scores.</p></div>
<div class="bc-glass bc-step"><i>3</i><h4>Look closer</h4><p>A flag means a human should review the scan. The heatmap shows what drove the score.</p></div></div>"""

NOTES_HTML = """<div class="bc-note"><b>How to read this.</b> A flag means <i>look closer</i>, not a diagnosis. A clear result does
not rule out a bleed: in our tests the model missed roughly 1 in 11 bleeds. The heatmap shows what drove the score, not where a bleed is;
it often highlighted the skull edge or scan background.<br>
<b>Data.</b> Trained on 6,650 slices of the TEKNOFEST-2021 stroke CT set; tested on a second public source (Kitamura, CC0). On a new source it
needs local checking before any real use.</div>"""


def verdict_html(p_bleed, cutoff):
    if p_bleed >= cutoff:
        cls, title = "flag", "Possible bleed flagged"
        text = "The bleed score is above your threshold. This is a prompt to look closer, not a diagnosis."
    else:
        cls, title = "clear", "No bleed flagged"
        text = "The bleed score is below your threshold. This does not rule out a bleed."
    return (f'<div class="bc-verdict {cls}"><div class="bc-ring" style="--p:{p_bleed * 100:.0f}"><b>{p_bleed:.2f}</b></div>'
            f'<div><h3>{title}</h3><p>Bleed score {p_bleed:.2f} (threshold {cutoff:.2f}). {text}</p></div></div>')


def bars_html(scores):
    rows = ""
    for name, v in scores.items():
        main = " main" if name == "Bleeding" else ""
        rows += (f'<div class="bc-row"><span>{name}</span>'
                 f'<div class="bc-track"><div class="bc-fill{main}" style="width:{v * 100:.1f}%"></div></div>'
                 f'<span class="bc-num">{v * 100:.0f}%</span></div>')
    return f'<div class="bc-glass bc-bars">{rows}</div>'

def analyse(img, cutoff):
    if img is None:
        return EMPTY_HTML, "", None
    img = img.convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    cam, p = gradcam(x)
    g = np.asarray(img.convert("L").resize((224, 224)), dtype=np.float32) / 255
    heat = cm.jet(cam)[..., :3]
    over = (0.6 * np.stack([g] * 3, -1) + 0.4 * heat).clip(0, 1)
    over = Image.fromarray((over * 255).astype("uint8")).resize((448, 448))
    scores = {c: float(p[i]) for i, c in enumerate(CLASSES)}
    return verdict_html(float(p[1]), cutoff), bars_html(scores), over


with gr.Blocks(title="BleedCheck CT") as demo:
    gr.HTML("<style>" + CSS + "</style>", elem_classes="bc-bare")
    gr.HTML(HERO_HTML, elem_classes="bc-bare")
    gr.HTML(STATS_HTML, elem_classes="bc-bare")
    gr.HTML(WARN_HTML, elem_classes="bc-bare")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Head CT slice", height=380)
            cutoff = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Flag threshold (lower = flags more scans)")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=1):
            verdict = gr.HTML(EMPTY_HTML, elem_classes="bc-bare")
            bars = gr.HTML("", elem_classes="bc-bare")
            heat = gr.Image(label="Where the model looked (Grad-CAM)", height=380, interactive=False)
    examples_dir = "examples"
    if os.path.isdir(examples_dir):
        files = sorted(os.path.join(examples_dir, f) for f in os.listdir(examples_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if files:
            gr.Examples(examples=[[f, 0.5] for f in files], inputs=[inp, cutoff], outputs=[verdict, bars, heat],
                        fn=analyse, run_on_click=True, cache_examples=False,
                        label="Public example scans (Kitamura, CC0). Click one to try it.")
    gr.HTML(STEPS_HTML, elem_classes="bc-bare")
    gr.HTML(NOTES_HTML, elem_classes="bc-bare")
    btn.click(analyse, [inp, cutoff], [verdict, bars, heat])

if __name__ == "__main__":
    demo.launch()
