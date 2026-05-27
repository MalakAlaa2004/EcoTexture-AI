"""
EcoTexture AI — Smart Waste Scanner
Bilingual AR/EN · Student-friendly · Egypt-focused
"""
from __future__ import annotations

import base64, random, time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from ecotexture_ai.config import (
    CO2_SAVINGS_KG, RECYCLING_BINS, WASTE_CLASSES, ensure_dirs,
)
from ecotexture_ai.explain import draw_sift_keypoints
from ecotexture_ai.predict import load_assets, predict_image
from ecotexture_ai.recommendations import (
    get_arabic_label, get_co2_savings, get_recommendation,
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoTexture AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)
ensure_dirs()

# ─── IMAGE HELPERS ──────────────────────────────────────────────────────────
ARTIFACTS = Path("C:/Users/LENOVO/.gemini/antigravity-ide/brain/d5b5fc0a-ae12-41be-8765-88bccf79b2b1")

def img_to_b64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""

HERO_B64    = img_to_b64(ARTIFACTS / "egypt_recycling_hero_1779913202507.png")
PLASTIC_B64 = img_to_b64(ARTIFACTS / "recycled_plastic_products_1779913151641.png")
PAPER_B64   = img_to_b64(ARTIFACTS / "recycled_paper_products_1779913164505.png")
GLASS_B64   = img_to_b64(ARTIFACTS / "recycled_glass_metal_1779913177606.png")

def b64_img(b64: str, alt: str = "", cls: str = "") -> str:
    if not b64:
        return ""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" class="{cls}" style="width:100%;border-radius:16px;object-fit:cover;">'

# ─── DATA ───────────────────────────────────────────────────────────────────
BIN_META = {
    "blue":   {"emoji":"🔵", "en":"Blue Recycling Bin",    "ar":"الصندوق الأزرق",      "color":"#3b82f6"},
    "green":  {"emoji":"🟢", "en":"Green Bin",             "ar":"الصندوق الأخضر",      "color":"#22c55e"},
    "yellow": {"emoji":"🟡", "en":"Yellow Recycling Bin",  "ar":"الصندوق الأصفر",      "color":"#eab308"},
    "brown":  {"emoji":"🟤", "en":"Brown Compost Bin",     "ar":"صندوق الكومبوست",     "color":"#a16207"},
    "black":  {"emoji":"⚫", "en":"General Waste Bin",     "ar":"النفايات العامة",     "color":"#6b7280"},
    "red":    {"emoji":"🔴", "en":"Hazardous Waste",       "ar":"نفايات خطرة",         "color":"#ef4444"},
    "orange": {"emoji":"🟠", "en":"E-Waste Collection",    "ar":"نفايات إلكترونية",    "color":"#f97316"},
    "purple": {"emoji":"🟣", "en":"Textile Collection",    "ar":"تجميع الملابس",       "color":"#a855f7"},
    "grey":   {"emoji":"⬜", "en":"Check Local Guidance",  "ar":"راجع الإرشادات",      "color":"#9ca3af"},
}

CLASS_ICON = {
    "Cardboard":"📦","Glass":"🍾","Metal":"🥫","Organic":"🌿","Paper":"📄",
    "Plastic":"🧴","Plastic_PET":"🧴","Plastic_HDPE":"🧴","Plastic_PVC":"⚠️",
    "Styrofoam":"🫧","Textile":"👕","E-Waste":"📱","Hazardous":"☠️","Trash":"🗑️",
}

FACTS = [
    ("🌊","Plastic takes 450 years to break down in the ocean!","البلاستيك يحتاج 450 عامًا ليتحلل في المحيط!"),
    ("⚡","1 recycled aluminium can = energy for 3 hours of TV!","علبة ألمنيوم واحدة = طاقة تلفاز لـ 3 ساعات!"),
    ("🌳","Recycling 1 tonne of paper saves 17 trees!","تدوير طن من الورق ينقذ 17 شجرة!"),
    ("💧","Glass can be recycled endlessly without losing quality!","الزجاج يُعاد تدويره للأبد دون فقدان جودته!"),
    ("🌍","Egypt generates 21 million tonnes of waste every year!","مصر تنتج 21 مليون طن قمامة سنويًا!"),
    ("♻️","Only 4% of Egypt's waste is formally recycled.","فقط 4% من قمامة مصر يُعاد تدويره رسميًا."),
    ("🌱","Composting cuts landfill methane emissions by 50%!","الكومبوست يقلل الميثان من المكبات بنسبة 50%!"),
]

UPCYCLE = {
    "Cardboard": [("🎨","Pencil organiser / منظم أقلام"),("🏠","Model house / نموذج بيت"),("🎁","Gift wrapping / تغليف هدايا")],
    "Glass":     [("🌸","Flower vase / إناء زهور"),("🕯️","Candle holder / حامل شمعة"),("🌿","Herb pot / وعاء أعشاب")],
    "Metal":     [("✏️","Pencil cup / كوب أقلام"),("🪴","Mini planter / وعاء نباتات"),("🎵","Music shaker / شيكر موسيقي")],
    "Paper":     [("🦢","Origami art / فن الأوريغامي"),("🎨","Sketchbook / دفتر رسم"),("🔖","Bookmarks / علامات كتب")],
    "Plastic_PET":[("🐦","Bird feeder / موزع طعام طيور"),("🌻","Planter pot / وعاء نباتات"),("💰","Piggy bank / كوزة توفير")],
    "Plastic_HDPE":[("🧺","Storage box / صندوق تخزين"),("🌻","Plant pot / وعاء نبات"),("🎡","Outdoor toy / لعبة خارجية")],
    "Plastic":   [("🐦","Bird feeder / موزع طعام طيور"),("🌻","Planter pot / وعاء نباتات"),("💰","Piggy bank / كوزة توفير")],
    "Textile":   [("🧸","Stuffed toy / لعبة محشوة"),("👜","Reusable bag / حقيبة قماش"),("🎀","Decoration / زينة")],
    "Organic":   [("🌱","Compost / سماد"),("🪱","Worm farm / مزرعة ديدان"),("🌺","Garden fertiliser / سماد حديقة")],
    "Trash":     [("🔁","Reduce first! / قلّل أولاً!"),("🛍️","Use reusable bags / استخدم أكياسًا قماشية"),("🌿","Go green daily / كن أخضر يوميًا")],
}

BIN_GUIDE = [
    ("blue",   "📦 Cardboard · 📄 Paper"),
    ("green",  "🍾 Glass"),
    ("yellow", "🥫 Metal · 🧴 Plastic PET · Plastic HDPE"),
    ("brown",  "🌿 Organic / Food Waste"),
    ("orange", "📱 E-Waste / Electronics"),
    ("purple", "👕 Textiles / Clothes"),
    ("red",    "☠️ Hazardous · ⚠️ PVC"),
    ("black",  "🫧 Styrofoam · 🗑️ General Trash"),
]

RECYCLED_IMGS = [
    (PLASTIC_B64, "♻️ Products from Recycled Plastic", "منتجات من البلاستيك المُعاد تدويره"),
    (PAPER_B64,   "📄 Products from Recycled Paper",   "منتجات من الورق المُعاد تدويره"),
    (GLASS_B64,   "🍾 Upcycled Glass & Metal",         "إبداعات من الزجاج والمعدن"),
]

# ─── SESSION STATE ───────────────────────────────────────────────────────────
for k, v in [("lang","en"),("page","scan"),("scans",0),("fi",random.randint(0,6))]:
    if k not in st.session_state:
        st.session_state[k] = v

lang = st.session_state.lang
ar   = (lang == "ar")

# ─── MASTER CSS (targets Streamlit internals correctly) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=Cairo:wght@400;600;700;800&display=swap');

/* ── GLOBAL ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Cairo', sans-serif !important;
}
.stApp {
    background: radial-gradient(ellipse at top left, #0d1b12 0%, #060d0a 60%, #000 100%) !important;
    min-height: 100vh;
}
.block-container {
    padding: 0.5rem 1rem 4rem !important;
    max-width: 580px !important;
    margin: 0 auto !important;
}
/* hide default label on file uploader */
[data-testid="stFileUploader"] label { display:none !important; }

/* ── KILL ALL default st colors ── */
p, li, span, h1, h2, h3, h4, div { color: rgba(255,255,255,0.88) !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── BUTTONS — nuke default, apply ours ── */
.stButton > button {
    width: 100% !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.7rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: rgba(34,197,94,0.18) !important;
    border-color: rgba(34,197,94,0.5) !important;
    color: #4ade80 !important;
    transform: translateY(-1px) !important;
}
/* active page button */
.stButton > button:focus {
    background: rgba(34,197,94,0.22) !important;
    border-color: #22c55e !important;
    color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.25) !important;
}
/* active LANG button */
button[kind="primary"] {
    background: rgba(34,197,94,0.22) !important;
    border-color: #22c55e !important;
    color: #4ade80 !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] section {
    background: rgba(34,197,94,0.05) !important;
    border: 2px dashed rgba(34,197,94,0.35) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(34,197,94,0.7) !important;
    background: rgba(34,197,94,0.09) !important;
}
[data-testid="stFileUploader"] section small {
    color: rgba(255,255,255,0.4) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── EXPANDER ── */
details {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
    padding: 0.2rem 0 !important;
    margin: 0.5rem 0 !important;
}
summary {
    padding: 0.7rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: rgba(255,255,255,0.75) !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-color: #22c55e transparent transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.6rem 0 0.6rem;">
  <div style="font-size:3.2rem;line-height:1;margin-bottom:0.3rem;">♻️</div>
  <div style="font-size:1.8rem;font-weight:900;
       background:linear-gradient(135deg,#4ade80,#22d3ee);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
       margin-bottom:0.2rem;">EcoTexture AI</div>
  <div style="font-size:0.75rem;letter-spacing:0.12em;text-transform:uppercase;
       color:rgba(255,255,255,0.35);font-weight:600;">
    Smart Waste Scanner · مسح النفايات الذكي
  </div>
</div>
""", unsafe_allow_html=True)

# ─── LANGUAGE ROW ────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("🇬🇧  English", key="lb_en",
                 type="primary" if lang=="en" else "secondary"):
        st.session_state.lang = "en"; st.rerun()
with c2:
    if st.button("🇪🇬  العربية", key="lb_ar",
                 type="primary" if lang=="ar" else "secondary"):
        st.session_state.lang = "ar"; st.rerun()
with c3:
    pass

# ─── NAV ROW ─────────────────────────────────────────────────────────────────
n1, n2, n3 = st.columns([1,1,1])
pages = {
    "scan": ("📷 Scanner","📷 مسح"),
    "learn":("📚 Learn","📚 تعلّم"),
    "about":("ℹ️ About","ℹ️ عن التطبيق"),
}
for (key,(en_lbl,ar_lbl)), col in zip(pages.items(), [n1,n2,n3]):
    with col:
        lbl = ar_lbl if ar else en_lbl
        if st.button(lbl, key=f"nav_{key}",
                     type="primary" if st.session_state.page==key else "secondary"):
            st.session_state.page = key; st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# Streak badge
if st.session_state.scans > 0:
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:0.6rem;
         background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.25);
         border-radius:99px;padding:0.35rem 1.2rem;margin:0 auto 0.8rem;width:fit-content;">
      <span style="font-size:1.2rem">🔥</span>
      <span style="font-size:1.15rem;font-weight:900;color:#fbbf24!important;">{st.session_state.scans}</span>
      <span style="font-size:0.75rem;color:rgba(255,255,255,0.5)!important;">
        {'items scanned today' if not ar else 'عناصر مسحتَها اليوم'}
      </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: SCANNER
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "scan":

    # Fun fact banner
    ico, en_f, ar_f = FACTS[st.session_state.fi % len(FACTS)]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(6,182,212,0.12),rgba(34,197,94,0.08));
         border:1px solid rgba(6,182,212,0.25);border-radius:16px;
         padding:0.9rem 1.1rem;margin-bottom:1rem;display:flex;gap:0.8rem;align-items:flex-start;">
      <span style="font-size:1.6rem;flex-shrink:0;">{ico}</span>
      <div>
        <div style="font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.9)!important;line-height:1.4;">
          {ar_f if ar else en_f}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # HERO image (Egypt illustration)
    if HERO_B64:
        st.markdown(
            f'<div style="border-radius:20px;overflow:hidden;margin-bottom:1rem;border:1px solid rgba(255,255,255,0.08);">'
            f'{b64_img(HERO_B64,"Egypt recycling")}</div>',
            unsafe_allow_html=True
        )

    # Upload prompt text
    st.markdown(f"""
    <div style="text-align:center;margin:0.5rem 0 0.3rem;">
      <div style="font-size:1.05rem;font-weight:700;color:rgba(255,255,255,0.9)!important;">
        {'📷 Scan your waste item' if not ar else '📷 امسح نفايتك'}
      </div>
      <div style="font-size:0.8rem;color:rgba(255,255,255,0.38)!important;margin-top:0.2rem;">
        {'Upload a photo — get instant recycling guidance' if not ar else 'ارفع صورة واحصل على إرشادات فورية'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload", type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        raw    = np.frombuffer(uploaded.read(), np.uint8)
        imgbgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)

        col_img, _ = st.columns([4,1])
        with col_img:
            st.image(imgrgb, use_container_width=True,
                     caption="📸 " + ("Your photo" if not ar else "صورتك"))

        with st.spinner("🤖 " + ("Analysing with AI…" if not ar else "تحليل بالذكاء الاصطناعي…")):
            t0 = time.perf_counter()
            try:
                model, centers, id2cls = load_assets()
                result = predict_image(imgbgr, model, centers, id2cls, lang=lang)
                ms = (time.perf_counter() - t0)*1000
                st.session_state.scans += 1
                st.session_state.fi   += 1

                label  = result["label"]
                conf   = result["confidence"]
                bcol   = result.get("bin_colour","grey")
                bm     = BIN_META.get(bcol, BIN_META["grey"])
                co2    = get_co2_savings(label)
                icon   = CLASS_ICON.get(label,"♻️")
                rec_en = get_recommendation(label,"en")
                rec_ar = get_recommendation(label,"ar")
                lab_ar = get_arabic_label(label)

                # ── BIG RESULT CARD ──────────────────────────────────────────
                pct = int(conf*100)
                st.markdown(f"""
                <div style="background:linear-gradient(145deg,{bm['color']}1a,{bm['color']}0d);
                     border:1.5px solid {bm['color']}55;border-radius:22px;
                     padding:1.6rem 1.4rem 1.2rem;margin:0.8rem 0;text-align:center;">

                  <!-- big icon -->
                  <div style="font-size:3.8rem;line-height:1;margin-bottom:0.4rem;">{bm['emoji']} {icon}</div>

                  <!-- class name -->
                  <div style="font-size:1.6rem;font-weight:900;
                       color:{bm['color']}!important;margin-bottom:0.1rem;">
                    {label.replace("_"," ")}
                  </div>
                  <div style="font-size:1rem;font-weight:700;
                       color:{bm['color']}99!important;direction:{'rtl' if ar else 'ltr'};">
                    {lab_ar}
                  </div>

                  <!-- bin name -->
                  <div style="font-size:0.82rem;font-weight:600;
                       color:rgba(255,255,255,0.5)!important;margin:0.5rem 0 0.8rem;">
                    {bm['ar'] if ar else bm['en']}
                  </div>

                  <!-- confidence bar -->
                  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">
                    <div style="flex:1;height:10px;border-radius:99px;background:rgba(255,255,255,0.09);">
                      <div style="height:10px;border-radius:99px;width:{pct}%;
                           background:linear-gradient(90deg,{bm['color']},{bm['color']}88);
                           transition:width 0.6s ease;"></div>
                    </div>
                    <span style="font-size:1rem;font-weight:800;color:{bm['color']}!important;min-width:46px;">
                      {pct}%
                    </span>
                  </div>

                  <!-- pills -->
                  <div style="display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap;">
                    <span style="background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.35);
                         border-radius:99px;padding:0.2rem 0.8rem;font-size:0.8rem;
                         font-weight:700;color:#4ade80!important;">🌿 CO₂ saved: {co2:.1f} kg</span>
                    <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.3);
                         border-radius:99px;padding:0.2rem 0.8rem;font-size:0.8rem;
                         font-weight:700;color:#22d3ee!important;">⚡ {ms:.0f} ms</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── WHERE TO PUT IT ──────────────────────────────────────────
                msg = rec_ar if ar else rec_en
                st.markdown(f"""
                <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.28);
                     border-radius:16px;padding:1rem 1.2rem;margin:0.5rem 0;">
                  <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
                       text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem;">
                    {'🗑️  Where to put it' if not ar else '🗑️  أين تضعه'}
                  </div>
                  <div style="font-size:0.95rem;font-weight:500;
                       color:rgba(255,255,255,0.88)!important;line-height:1.55;
                       direction:{'rtl' if ar else 'ltr'};">
                    {msg}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── UPCYCLE IDEAS ─────────────────────────────────────────────
                ideas = UPCYCLE.get(label, UPCYCLE.get("Trash"))
                st.markdown(f"""
                <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
                     text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.4rem;">
                  💡 {'Upcycle Ideas' if not ar else 'أفكار لإعادة الاستخدام'}
                </div>
                """, unsafe_allow_html=True)
                cols = st.columns(len(ideas))
                for col, (ico_i, lbl_i) in zip(cols, ideas):
                    parts = lbl_i.split(" / ")
                    txt = parts[1] if ar and len(parts)>1 else parts[0]
                    col.markdown(f"""
                    <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                         border-radius:14px;padding:0.75rem 0.5rem;text-align:center;">
                      <div style="font-size:1.6rem;">{ico_i}</div>
                      <div style="font-size:0.72rem;font-weight:600;
                           color:rgba(255,255,255,0.7)!important;margin-top:0.3rem;
                           line-height:1.3;">{txt}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # milestone
                if st.session_state.scans in [1,5,10,25,50]:
                    st.balloons()
                    st.success(f"🎉 {'Milestone! ' if not ar else 'إنجاز! '}{st.session_state.scans} {'scans!' if not ar else 'مسحات!'}")

                # ── AI EXPLAINABILITY (collapsed) ──────────────────────────
                with st.expander("🔬 " + ("AI Vision Explanation" if not ar else "شرح الذكاء الاصطناعي")):
                    cx1, cx2 = st.columns(2)
                    with cx1:
                        st.caption("GradCAM — " + ("Where AI looked" if not ar else "أين نظر الذكاء الاصطناعي"))
                        st.image(result["heatmap"], use_container_width=True)
                    with cx2:
                        st.caption("SIFT — " + ("Texture keypoints" if not ar else "نقاط الملمس"))
                        st.image(result["sift_overlay"], use_container_width=True)

                with st.expander("📊 " + ("Top predictions" if not ar else "أفضل التنبؤات")):
                    for item in result["top_k"]:
                        p = item["confidence"]*100
                        st.markdown(f"""
                        <div style="margin:0.3rem 0;">
                          <div style="font-size:0.82rem;color:rgba(255,255,255,0.7)!important;">
                            {CLASS_ICON.get(item['label'],'♻️')} {item['label'].replace('_',' ')} — <b>{p:.1f}%</b>
                          </div>
                          <div style="background:rgba(255,255,255,0.07);border-radius:99px;height:5px;margin-top:3px;">
                            <div style="background:#22c55e;border-radius:99px;height:5px;width:{p:.1f}%;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

            except FileNotFoundError as exc:
                st.warning(f"⚠️ Model not found: {exc}")
                st.image(draw_sift_keypoints(imgrgb), caption="SIFT Preview", use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: LEARN
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "learn":

    # Hero stats
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
         text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">
         🌍 {'Egypt Waste Crisis' if not ar else 'أزمة النفايات في مصر'}
    </div>
    """, unsafe_allow_html=True)

    stats = [
        ("🏙️","21 مليون طن","21M tonnes","تُنتج في مصر سنويًا","generated in Egypt yearly"),
        ("♻️","4–5% فقط","Only 4–5%","يُعاد تدويره رسميًا","formally recycled"),
        ("🌊","مليون طن بلاستيك","1M tonnes plastic","تدخل النيل سنويًا","enter the Nile yearly"),
        ("🌿","55–60%","55–60%","من النفايات عضوية قابلة للكومبوست","of waste is organic & compostable"),
        ("👷","الزبالون","The Zabbaleen","يعيدون تدوير 80% مما يجمعونه","recycle 80% of what they collect"),
    ]
    for ico, ar_n, en_n, ar_d, en_d in stats:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:0.9rem;
             background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
             border-radius:16px;padding:0.9rem 1rem;margin:0.4rem 0;">
          <span style="font-size:1.5rem;flex-shrink:0;">{ico}</span>
          <div>
            <div style="font-size:1.1rem;font-weight:800;color:#4ade80!important;">{ar_n if ar else en_n}</div>
            <div style="font-size:0.85rem;color:rgba(255,255,255,0.6)!important;">{ar_d if ar else en_d}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Recycled product IMAGES
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
         text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;">
         ♻️ {'What Gets Made from Recycled Waste' if not ar else 'ماذا يُصنع من النفايات المُعادة تدويرها؟'}
    </div>
    """, unsafe_allow_html=True)

    for b64, en_cap, ar_cap in RECYCLED_IMGS:
        if b64:
            cap = ar_cap if ar else en_cap
            st.markdown(f"""
            <div style="border-radius:18px;overflow:hidden;margin-bottom:0.8rem;
                 border:1px solid rgba(255,255,255,0.1);">
              {b64_img(b64, cap)}
              <div style="background:rgba(0,0,0,0.6);padding:0.5rem 0.9rem;
                   font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.85)!important;
                   text-align:center;">{cap}</div>
            </div>""", unsafe_allow_html=True)

    # Bin colour guide
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
         text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;">
         🗑️ {'Bin Colour Guide' if not ar else 'دليل ألوان الصناديق'}
    </div>""", unsafe_allow_html=True)

    for colour, items in BIN_GUIDE:
        bm = BIN_META[colour]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;
             border-left:3px solid {bm['color']};
             background:rgba(255,255,255,0.03);border-radius:0 12px 12px 0;
             padding:0.6rem 0.9rem;margin:0.3rem 0;">
          <span style="font-size:1.2rem;">{bm['emoji']}</span>
          <div>
            <span style="font-size:0.88rem;font-weight:700;color:{bm['color']}!important;">
              {bm['ar'] if ar else bm['en']}
            </span>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.4)!important;margin-top:1px;">
              {items}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Fun facts
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
         text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;">
         ⚡ {'Did You Know?' if not ar else 'هل تعلم؟'}
    </div>""", unsafe_allow_html=True)

    for ico, en_f, ar_f in FACTS:
        st.markdown(f"""
        <div style="display:flex;gap:0.75rem;align-items:flex-start;
             background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.18);
             border-radius:14px;padding:0.8rem 1rem;margin:0.35rem 0;">
          <span style="font-size:1.3rem;flex-shrink:0;">{ico}</span>
          <div style="font-size:0.87rem;font-weight:500;
               color:rgba(255,255,255,0.82)!important;line-height:1.5;">
            {ar_f if ar else en_f}
          </div>
        </div>""", unsafe_allow_html=True)

    # Upcycle ideas by material
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:700;color:#4ade80!important;
         text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;">
         💡 {'Create from Waste' if not ar else 'ابتكر من النفايات'}
    </div>""", unsafe_allow_html=True)

    for mat, ideas in UPCYCLE.items():
        ico_mat = CLASS_ICON.get(mat,"♻️")
        with st.expander(f"{ico_mat} {mat.replace('_',' ')}"):
            for ico_i, lbl_i in ideas:
                parts = lbl_i.split(" / ")
                txt_en, txt_ar = parts[0], (parts[1] if len(parts)>1 else parts[0])
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;">
                  <span style="font-size:1.1rem;">{ico_i}</span>
                  <span style="font-size:0.88rem;color:rgba(255,255,255,0.8)!important;">
                    {txt_ar if ar else txt_en}
                  </span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "about":
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
         border-radius:22px;padding:2rem 1.5rem;text-align:center;margin-top:0.5rem;">

      <div style="font-size:3rem;margin-bottom:0.6rem;">🤖♻️</div>

      <div style="font-size:1.4rem;font-weight:800;
           background:linear-gradient(135deg,#4ade80,#22d3ee);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           margin-bottom:1rem;">EcoTexture AI</div>

      <div style="font-size:0.88rem;color:rgba(255,255,255,0.6)!important;
           line-height:1.9;margin-bottom:1.2rem;">
        {'Hybrid AI waste classifier for Egyptian students' if not ar else 'مصنّف ذكاء اصطناعي هجين لطلاب مصر'}<br>
        <br>
        <b style="color:rgba(255,255,255,0.8)!important;">{'Architecture:' if not ar else 'المعمارية:'}</b>
        EfficientNetB0 + SIFT + Cross-Attention<br>
        <b style="color:rgba(255,255,255,0.8)!important;">{'Best Validation Acc:' if not ar else 'أفضل دقة:'}</b>
        89.38%<br>
        <b style="color:rgba(255,255,255,0.8)!important;">{'Trainable Params:' if not ar else 'المعاملات القابلة للتدريب:'}</b>
        830K (edge-ready)<br>
        <b style="color:rgba(255,255,255,0.8)!important;">{'Dataset:' if not ar else 'مجموعة البيانات:'}</b>
        TrashNet + TACO
      </div>

      <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
           border-radius:14px;padding:1rem;margin-bottom:1rem;">
        <div style="font-size:0.78rem;font-weight:700;color:#4ade80!important;
             text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">
          👩‍💻 {'Team' if not ar else 'الفريق'}
        </div>
        <div style="font-size:0.9rem;color:rgba(255,255,255,0.8)!important;">
          <b>Malak Alaa</b> — {'Author & Engineer' if not ar else 'مؤلفة ومهندسة'}<br>
          <b>Dr. Islam Gamal</b> — {'Supervisor' if not ar else 'المشرف'}
        </div>
      </div>

      <div style="font-size:0.82rem;color:rgba(255,255,255,0.5)!important;line-height:1.8;">
        {'Department of Artificial Intelligence' if not ar else 'قسم الذكاء الاصطناعي'}<br>
        {'Faculty of Computer Science & AI' if not ar else 'كلية علوم الحاسب والذكاء الاصطناعي'}<br>
        {'Capital University' if not ar else 'جامعة العاصمة'}
      </div>

      <div style="margin-top:1rem;display:flex;gap:0.4rem;justify-content:center;flex-wrap:wrap;">
        {"".join(f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:99px;padding:0.2rem 0.7rem;font-size:0.72rem;color:rgba(255,255,255,0.5)!important;">{t}</span>' for t in ["UGRF 2026","WISE Award","SDG 4","SDG 12","SDG 13"])}
      </div>
    </div>
    """, unsafe_allow_html=True)
