"""
app.py
------
Streamlit web interface for the Phishing Email Detector.
Run with: streamlit run app.py
"""

import sys
import joblib
import streamlit as st

from utils.paths import MODEL_PATH, VECTORIZER_PATH

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin: 1.2rem 0;
    backdrop-filter: blur(12px);
}

/* ── Result banners ── */
.result-phishing {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.3));
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(21,128,61,0.3));
    border: 1px solid rgba(34,197,94,0.5);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-title {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.result-sub {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Confidence meter label ── */
.conf-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}

/* ── Tip box ── */
.tip-box {
    background: rgba(99,102,241,0.12);
    border-left: 3px solid #818cf8;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    color: #c7d2fe;
    font-size: 0.88rem;
    margin-top: 1rem;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load model (cached so it only loads once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ Phishing Email Detector</h1>
    <p>Paste any email below and the AI will classify it as Safe or Phishing in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model missing warning
# ──────────────────────────────────────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ **Model not found.** Please run `python train_model.py` first to "
        "train and save the model, then restart this app."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Input card
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("#### 📧 Email Content")

email_text = st.text_area(
    label="email_input",
    label_visibility="collapsed",
    placeholder="Paste the full email text here…\n\nE.g. Dear user, your account has been suspended. Click here to verify your identity immediately.",
    height=220,
    key="email_input",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    analyse = st.button("🔍 Analyse Email", use_container_width=True, type="primary")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["email_input"] = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
if analyse:
    if not email_text.strip():
        st.warning("⚠️ Please paste some email text before analysing.")
    else:
        with st.spinner("Analysing…"):
            vec        = vectorizer.transform([email_text])
            prediction = model.predict(vec)[0]
            proba      = model.predict_proba(vec)[0]
            confidence = max(proba) * 100
            phish_prob = proba[1] * 100
            safe_prob  = proba[0] * 100

        # ── Result banner ──────────────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-phishing">
                <div style="font-size:2.8rem">🚨</div>
                <div class="result-title" style="color:#f87171">Phishing Email</div>
                <p class="result-sub">This email shows signs of a phishing attempt. Do not click any links.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:2.8rem">✅</div>
                <div class="result-title" style="color:#4ade80">Safe Email</div>
                <p class="result-sub">No phishing indicators detected. This email appears legitimate.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence metrics ─────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Confidence Breakdown")

        c1, c2, c3 = st.columns(3)
        c1.metric("Model Confidence", f"{confidence:.1f}%")
        c2.metric("🚨 Phishing Probability", f"{phish_prob:.1f}%")
        c3.metric("✅ Safe Probability",     f"{safe_prob:.1f}%")

        st.markdown('<p class="conf-label">Phishing risk meter</p>', unsafe_allow_html=True)
        st.progress(int(phish_prob))

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Safety tips ────────────────────────────────────────────────────
        if prediction == 1:
            st.markdown("""
            <div class="tip-box">
            <strong>🔒 Safety Tips:</strong><br>
            • Do not click any links or download attachments from this email.<br>
            • Do not enter any personal or financial information.<br>
            • Report it to your IT/security team or email provider.<br>
            • Delete the email from your inbox.
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem'>"
    "Built with Scikit-learn · TF-IDF · Streamlit &nbsp;|&nbsp; "
    "Thiranex Cybersecurity Internship · Vinayak D"
    "</p>",
    unsafe_allow_html=True,
)
