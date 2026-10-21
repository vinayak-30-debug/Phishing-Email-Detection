"""
app.py
------
Streamlit web interface for the Phishing Email Detector.
Run with: streamlit run app.py
"""

import sys
import numpy as np
import joblib
import streamlit as st

from utils.paths import MODEL_PATH, VECTORIZER_PATH

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
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

/* ── Keyword Heatmap ── */
.heatmap-section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.6rem;
}
.heatmap-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.kw-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: default;
}
.kw-pill:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.kw-pill-phish {
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(185,28,28,0.15));
    color: #fca5a5;
    border-color: rgba(239,68,68,0.35);
}
.kw-pill-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.25), rgba(21,128,61,0.15));
    color: #86efac;
    border-color: rgba(34,197,94,0.35);
}
.kw-bar-container {
    width: 50px;
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: hidden;
}
.kw-bar-phish {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #f87171, #ef4444);
}
.kw-bar-safe {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #4ade80, #22c55e);
}
.kw-legend {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.8rem;
    font-size: 0.78rem;
    color: #94a3b8;
}
.kw-legend-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.3rem;
    vertical-align: middle;
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

        # ── Keyword Heatmap ────────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Keyword Heatmap")
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.88rem;margin-top:-0.5rem">' 
            'Words & phrases that most influenced the prediction</p>',
            unsafe_allow_html=True,
        )

        # Get feature names and weights from the model
        feature_names = np.array(vectorizer.get_feature_names_out())
        email_tfidf = vec.toarray().flatten()

        # Get model coefficients / feature importances
        if hasattr(model, 'coef_'):
            # Linear models (LogisticRegression, NaiveBayes, etc.)
            coefs = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models (RandomForest, etc.)
            coefs = model.feature_importances_
        else:
            coefs = np.zeros(len(feature_names))

        # Combine: importance = tfidf_weight * |model_coefficient|
        # This highlights terms that are BOTH present in the email AND important to the model
        influence = email_tfidf * coefs

        # Get top phishing-leaning terms (positive influence) and safe-leaning terms (negative)
        nonzero_mask = email_tfidf > 0  # only terms present in this email
        active_indices = np.where(nonzero_mask)[0]

        if len(active_indices) > 0:
            # Get top positive-influence (phishing) and negative-influence (safe) terms
            phish_sorted = sorted(active_indices, key=lambda i: influence[i], reverse=True)
            safe_sorted = sorted(active_indices, key=lambda i: influence[i])

            top_phish = [(feature_names[i], influence[i]) for i in phish_sorted if influence[i] > 0][:12]
            top_safe = [(feature_names[i], abs(influence[i])) for i in safe_sorted if influence[i] < 0][:8]

            max_influence = max(
                max((v for _, v in top_phish), default=0),
                max((v for _, v in top_safe), default=0),
                1e-10  # avoid division by zero
            )

            # ── Render phishing keywords ──
            if top_phish:
                st.markdown(
                    '<p class="heatmap-section-title">🚨 Phishing Indicators</p>',
                    unsafe_allow_html=True,
                )
                pills_html = '<div class="heatmap-container">'
                for word, score in top_phish:
                    pct = min(int((score / max_influence) * 100), 100)
                    pills_html += (
                        f'<div class="kw-pill kw-pill-phish">'
                        f'<span>{word}</span>'
                        f'<div class="kw-bar-container">'
                        f'<div class="kw-bar-phish" style="width:{pct}%"></div>'
                        f'</div></div>'
                    )
                pills_html += '</div>'
                st.markdown(pills_html, unsafe_allow_html=True)

            # ── Render safe keywords ──
            if top_safe:
                st.markdown(
                    '<p class="heatmap-section-title" style="margin-top:1rem">'
                    '✅ Safety Indicators</p>',
                    unsafe_allow_html=True,
                )
                pills_html = '<div class="heatmap-container">'
                for word, score in top_safe:
                    pct = min(int((score / max_influence) * 100), 100)
                    pills_html += (
                        f'<div class="kw-pill kw-pill-safe">'
                        f'<span>{word}</span>'
                        f'<div class="kw-bar-container">'
                        f'<div class="kw-bar-safe" style="width:{pct}%"></div>'
                        f'</div></div>'
                    )
                pills_html += '</div>'
                st.markdown(pills_html, unsafe_allow_html=True)

            # ── Legend ──
            st.markdown(
                '<div class="kw-legend">'
                '<span><span class="kw-legend-dot" style="background:#ef4444"></span>'
                'Phishing signal &mdash; pushes toward phishing</span>'
                '<span><span class="kw-legend-dot" style="background:#22c55e"></span>'
                'Safety signal &mdash; pushes toward safe</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No keyword data available for this email.")

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
