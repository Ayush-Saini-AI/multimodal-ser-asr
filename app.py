"""
Emotion-Aware Speech Transcription — Streamlit Frontend
Run with: streamlit run app.py
Tested on: Windows, Python 3.9+, CPU only
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import soundfile as sf
import tempfile
import os
import re
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Emotion-Aware Speech Analyzer",
    page_icon="🎙️",
    layout="wide",
)

# ── Minimal clean CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .metric-box.red   { border-left-color: #ef4444; }
    .metric-box.green { border-left-color: #22c55e; }
    .metric-box.amber { border-left-color: #f59e0b; }
    .metric-box.grey  { border-left-color: #9ca3af; }
    .metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.15rem; font-weight: 600; color: #111827; margin-top: 2px; }
    .report-box {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .insight-banner {
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-size: 1rem;
        font-weight: 500;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #4f8ef7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover { background-color: #3b7de8; }
    h1 { color: #111827 !important; }
    h2, h3 { color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS  (cached so they load only once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_whisper():
    import whisper
    return whisper.load_model("tiny")


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,   # CPU
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(file_bytes, suffix):
    """Write uploaded bytes to a temp file and load with librosa."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
    os.unlink(tmp_path)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio, sr


def transcribe(audio, sr):
    """Transcribe audio array using Whisper tiny."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name
    model = load_whisper()
    result = model.transcribe(tmp_path, language="en", fp16=False)
    os.unlink(tmp_path)
    return result["text"].strip()


def get_sentiment(text):
    pipe = load_sentiment_pipeline()
    res = pipe(text[:512])[0]
    return res["label"], res["score"]


def extract_acoustic(audio, sr):
    """Extract all acoustic features. Returns a dict."""
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
    pitch_vals = []
    for t in range(pitches.shape[1]):
        mc = magnitudes[:, t]
        if mc.max() > 0:
            idx = mc.argmax()
            p = pitches[idx, t]
            if p > 50:
                pitch_vals.append(p)
    pitch_vals = np.array(pitch_vals) if pitch_vals else np.array([0.0])

    # Energy
    rms = librosa.feature.rms(y=audio)[0]

    # Tempo
    tempo_result = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = tempo_result[0]
    tempo_val = float(np.mean(tempo)) if hasattr(tempo, "__len__") else float(tempo)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    return {
        "mfccs":        mfccs,
        "mfcc_mean":    np.mean(mfccs, axis=1),
        "pitch_vals":   pitch_vals,
        "pitch_mean":   float(np.mean(pitch_vals)),
        "pitch_std":    float(np.std(pitch_vals)),
        "pitch_range":  float(np.ptp(pitch_vals)),
        "rms":          rms,
        "energy_mean":  float(np.mean(rms)),
        "energy_max":   float(np.max(rms)),
        "tempo":        tempo_val,
        "zcr_mean":     float(np.mean(zcr)),
        "spec_centroid": spec_centroid,
        "spec_cent_mean": float(np.mean(spec_centroid)),
    }


def generate_insight(feat, sentiment_label, sentiment_score, transcript):
    """Rule-based fusion insight engine."""
    flags, details = [], []

    energy = feat["energy_mean"]
    arousal = "HIGH" if energy > 0.05 else ("MEDIUM" if energy > 0.02 else "LOW")
    if arousal == "HIGH":
        details.append("high vocal energy")
    elif arousal == "MEDIUM":
        details.append("moderate vocal energy")
    else:
        details.append("low vocal energy")

    if feat["pitch_mean"] > 250:
        flags.append("ELEVATED_PITCH")
        details.append("elevated pitch")
    elif 0 < feat["pitch_mean"] < 100:
        flags.append("LOW_PITCH")
        details.append("low pitch")

    if feat["tempo"] > 140:
        flags.append("FAST_SPEECH")
        details.append("fast speech rate")
    elif feat["tempo"] < 70:
        flags.append("SLOW_SPEECH")
        details.append("slow speech rate")

    valence = "POSITIVE" if sentiment_label == "POSITIVE" else "NEGATIVE"

    conflict = (valence == "POSITIVE" and arousal == "HIGH"
                and sentiment_score < 0.80 and "ELEVATED_PITCH" in flags)
    if conflict:
        flags.append("SENTIMENT_ACOUSTIC_CONFLICT")

    frustration_kw = ["frustrated","angry","terrible","awful","useless",
                      "horrible","worst","broken","nothing","never","stupid"]
    hits = [w for w in frustration_kw if w in transcript.lower()]
    if hits:
        flags.append("FRUSTRATION_KEYWORDS")
        details.append(f"frustration keywords: {hits}")

    # Final category
    if "FRUSTRATION_KEYWORDS" in flags or (valence == "NEGATIVE" and arousal == "HIGH"):
        emotion   = "FRUSTRATION / ANGER"
        insight   = "Speaker appears frustrated or angry — potential escalation risk."
        color     = "red"
        emoji     = "⚠️"
    elif conflict:
        emotion   = "POSSIBLE SARCASM"
        insight   = "Sentiment-acoustic conflict detected — speaker may be sarcastic."
        color     = "amber"
        emoji     = "🔍"
    elif valence == "POSITIVE" and arousal == "HIGH":
        emotion   = "ENTHUSIASM"
        insight   = "Speaker sounds enthusiastic and engaged."
        color     = "green"
        emoji     = "✅"
    elif valence == "POSITIVE" and arousal == "LOW":
        emotion   = "CALM SATISFACTION"
        insight   = "Speaker appears calm and satisfied."
        color     = "green"
        emoji     = "✅"
    elif valence == "NEGATIVE" and arousal == "LOW":
        emotion   = "DISAPPOINTMENT"
        insight   = "Speaker sounds disappointed or disengaged."
        color     = "amber"
        emoji     = "ℹ️"
    else:
        emotion   = "NEUTRAL"
        insight   = "No strong emotion detected — tone appears neutral."
        color     = "grey"
        emoji     = "ℹ️"

    return {
        "emotion": emotion, "insight": insight,
        "color": color, "emoji": emoji,
        "arousal": arousal, "valence": valence,
        "flags": flags, "details": details,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_waveform_mfcc(audio, sr, feat):
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.35)
    fig.patch.set_facecolor("#f9fafb")

    duration  = librosa.get_duration(y=audio, sr=sr)
    time_axis = np.linspace(0, duration, len(audio))

    # Waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, audio, color="#4f8ef7", linewidth=0.6, alpha=0.85)
    ax1.fill_between(time_axis, audio, alpha=0.12, color="#4f8ef7")
    ax1.set_title("Waveform", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_facecolor("#fafafa")
    ax1.grid(True, alpha=0.2)

    # MFCC heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    img = librosa.display.specshow(feat["mfccs"], sr=sr, x_axis="time",
                                   ax=ax2, cmap="RdYlBu_r")
    fig.colorbar(img, ax=ax2, label="Value")
    ax2.set_title("MFCC Heatmap (13 coefficients)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("MFCC Index")

    # RMS Energy
    ax3 = fig.add_subplot(gs[1, 1])
    rms_time = librosa.times_like(feat["rms"], sr=sr)
    ax3.plot(rms_time, feat["rms"], color="#f59e0b", linewidth=1.2)
    ax3.fill_between(rms_time, feat["rms"], alpha=0.2, color="#f59e0b")
    ax3.axhline(y=feat["energy_mean"], color="black", linestyle="--",
                linewidth=1, label=f'Mean: {feat["energy_mean"]:.4f}')
    ax3.set_title("RMS Energy over Time", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RMS")
    ax3.legend(fontsize=8)
    ax3.set_facecolor("#fafafa")

    return fig


def plot_sentiment_bar(label, score):
    fig, ax = plt.subplots(figsize=(7, 1.6))
    fig.patch.set_facecolor("#f9fafb")
    color = "#22c55e" if label == "POSITIVE" else "#ef4444"
    ax.barh(["Confidence"], [score], color=color, height=0.45)
    ax.barh(["Confidence"], [1 - score], left=[score],
            color="#e5e7eb", height=0.45)
    ax.axvline(x=0.5, color="#374151", linestyle="--", linewidth=1,
               alpha=0.6, label="Decision boundary (0.5)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(f"DistilBERT Sentiment: {label}  ({score*100:.1f}%)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT STRING
# ══════════════════════════════════════════════════════════════════════════════

def build_report(transcript, sentiment_label, sentiment_score,
                 feat, ins):
    div = "═" * 52
    thin = "─" * 52
    emoji = "😊" if sentiment_label == "POSITIVE" else "😠"
    flags_str = "\n".join(f"  ⚠  {f}" for f in ins["flags"]) if ins["flags"] else "  ✅  None"

    return f"""{div}
  🎙️  EMOTION-AWARE SPEECH ANALYSIS REPORT
  Multimodal Fusion Pipeline — Review 1
{div}

📝 TRANSCRIPTION
{thin}
  {transcript}

🧠 SENTIMENT  (DistilBERT)
{thin}
  Label      : {emoji} {sentiment_label}
  Confidence : {sentiment_score:.4f}  ({sentiment_score*100:.1f}%)
  Valence    : {ins["valence"]}

🔬 ACOUSTIC FEATURES  (librosa)
{thin}
  Pitch (F0)   : {feat["pitch_mean"]:.1f} Hz  (std: {feat["pitch_std"]:.1f}, range: {feat["pitch_range"]:.1f})
  RMS Energy   : {feat["energy_mean"]:.5f}  (max: {feat["energy_max"]:.5f})
  Tempo        : {feat["tempo"]:.1f} BPM
  Spec.Centroid: {feat["spec_cent_mean"]:.1f} Hz
  ZCR          : {feat["zcr_mean"]:.5f}
  Arousal      : {ins["arousal"]}

⚑  SYSTEM FLAGS
{thin}
{flags_str}

💡 AI INSIGHT  (Rule-Based Fusion)
{thin}
  Emotion  : {ins["emotion"]}
  Evidence : {" | ".join(ins["details"]) if ins["details"] else "Standard baseline"}

  ➤  {ins["emoji"]}  {ins["insight"]}

{div}
  ✅  Pipeline Complete — Review 1 Milestone
{div}"""


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎙️ Emotion-Aware Speech Transcription")
st.caption("Multimodal Fusion Pipeline — Review 1 Demo | NLP Project")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
This app demonstrates a **multimodal speech analysis pipeline**:

- **Whisper tiny** → Speech to Text
- **DistilBERT** → Sentiment Analysis
- **librosa** → Acoustic Features
- **Rule-based Fusion** → AI Insight

**For Review 1** — unimodal pipelines  
**For Review 2** — neural fusion layer
""")
    st.markdown("---")
    st.markdown("### 📋 Pipeline Steps")
    steps = [
        "1. Upload Audio",
        "2. Load + Preprocess",
        "3. Whisper Transcription",
        "4. Text Sentiment",
        "5. Acoustic Features",
        "6. AI Insight Fusion",
        "7. Final Report",
    ]
    for s in steps:
        st.markdown(f"- {s}")
    st.markdown("---")
    st.markdown("### ⏱️ Est. Time")
    st.markdown("""
| Step | Time |
|---|---|
| First model load | ~3 min |
| Transcription | ~60–90s |
| Sentiment | ~15s |
| Features | ~10s |
""")

# ── Main: Upload ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📂 Step 1 — Upload Audio")
    uploaded = st.file_uploader(
        "Upload a .wav or .mp3 file",
        type=["wav", "mp3"],
        help="Record yourself speaking 1–2 sentences for the best demo output."
    )
    st.caption("💡 Tip: Record an angry or frustrated sentence for the most interesting results.")

    analyze_btn = st.button("▶  Analyze Audio", disabled=(uploaded is None))

with col_right:
    st.subheader("📌 What to expect")
    st.markdown("""
After clicking **Analyze Audio**, the app will:
1. Load and normalize your audio
2. Transcribe it with Whisper
3. Run sentiment analysis
4. Extract pitch, energy, tempo
5. Generate an AI insight
6. Show plots + final report

**First run:** ~3–4 min (model downloads)  
**Subsequent runs:** ~1–2 min (models cached)
""")

st.markdown("---")

# ── Analysis ─────────────────────────────────────────────────────────────────
if analyze_btn and uploaded is not None:

    file_bytes = uploaded.read()
    suffix     = ".wav" if uploaded.name.endswith(".wav") else ".mp3"

    # ── Step 1: Load audio ────────────────────────────────────────────────────
    with st.spinner("Loading and preprocessing audio..."):
        audio, sr = load_audio(file_bytes, suffix)
        duration  = librosa.get_duration(y=audio, sr=sr)

    st.success(f"✅ Audio loaded — {duration:.1f}s at {sr}Hz")

    # ── Step 2: Transcribe ────────────────────────────────────────────────────
    with st.spinner("Transcribing with Whisper tiny... (may take 60–90s on first run)"):
        transcript = transcribe(audio, sr)

    if not transcript:
        transcript = "I am really frustrated with this service. Nothing seems to work."
        st.warning("⚠️ No speech detected — using demo transcript for pipeline illustration.")

    st.markdown("### 📝 Transcript")
    st.info(f'"{transcript}"')

    # ── Step 3: Sentiment ─────────────────────────────────────────────────────
    with st.spinner("Running DistilBERT sentiment analysis..."):
        sent_label, sent_score = get_sentiment(transcript)

    # ── Step 4: Acoustic features ─────────────────────────────────────────────
    with st.spinner("Extracting acoustic features (MFCC, pitch, energy, tempo)..."):
        feat = extract_acoustic(audio, sr)

    # ── Step 5: Insight ───────────────────────────────────────────────────────
    ins = generate_insight(feat, sent_label, sent_score, transcript)

    st.markdown("---")

    # ── Results row 1: Metrics ────────────────────────────────────────────────
    st.subheader("📊 Results")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
<div class="metric-box {'green' if sent_label=='POSITIVE' else 'red'}">
  <div class="metric-label">Sentiment</div>
  <div class="metric-value">{'😊' if sent_label=='POSITIVE' else '😠'} {sent_label}</div>
</div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
<div class="metric-box">
  <div class="metric-label">Confidence</div>
  <div class="metric-value">{sent_score*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
<div class="metric-box">
  <div class="metric-label">Arousal Level</div>
  <div class="metric-value">{ins['arousal']}</div>
</div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
<div class="metric-box {ins['color']}">
  <div class="metric-label">Emotion</div>
  <div class="metric-value">{ins['emoji']} {ins['emotion']}</div>
</div>""", unsafe_allow_html=True)

    # ── Insight banner ────────────────────────────────────────────────────────
    banner_colors = {
        "red":   ("🔴", "#fef2f2", "#991b1b"),
        "green": ("🟢", "#f0fdf4", "#166534"),
        "amber": ("🟡", "#fffbeb", "#92400e"),
        "grey":  ("⚪", "#f9fafb", "#374151"),
    }
    _, bg, fg = banner_colors.get(ins["color"], banner_colors["grey"])
    st.markdown(f"""
<div class="insight-banner" style="background:{bg}; color:{fg}; border: 1px solid {fg}33;">
  <strong>💡 AI Insight:</strong> {ins['emoji']} {ins['insight']}
</div>""", unsafe_allow_html=True)

    if ins["flags"]:
        st.markdown(f"**⚑ Flags:** `{'` · `'.join(ins['flags'])}`")

    st.markdown("---")

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.subheader("📈 Waveform + MFCC + Energy")
    fig_main = plot_waveform_mfcc(audio, sr, feat)
    st.pyplot(fig_main, use_container_width=True)
    plt.close(fig_main)

    st.subheader("🧠 Sentiment Confidence")
    fig_sent = plot_sentiment_bar(sent_label, sent_score)
    st.pyplot(fig_sent, use_container_width=True)
    plt.close(fig_sent)

    # ── Acoustic feature table ────────────────────────────────────────────────
    st.subheader("🔬 Acoustic Feature Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Pitch Mean (F0)", f"{feat['pitch_mean']:.1f} Hz")
        st.metric("RMS Energy Mean", f"{feat['energy_mean']:.5f}")
        st.metric("Tempo", f"{feat['tempo']:.1f} BPM")
    with c2:
        st.metric("Pitch Std Dev", f"{feat['pitch_std']:.1f} Hz")
        st.metric("Spectral Centroid", f"{feat['spec_cent_mean']:.1f} Hz")
        st.metric("Zero Crossing Rate", f"{feat['zcr_mean']:.5f}")

    st.markdown("---")

    # ── Final Report ──────────────────────────────────────────────────────────
    st.subheader("📋 Final Report")
    report = build_report(transcript, sent_label, sent_score, feat, ins)
    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

    # Download button
    st.download_button(
        label="⬇️  Download Report as .txt",
        data=report,
        file_name="emotion_speech_report.txt",
        mime="text/plain",
    )

elif not analyze_btn:
    st.markdown("""
<div style="text-align:center; padding: 3rem; color: #9ca3af;">
  <div style="font-size: 3rem;">🎙️</div>
  <div style="font-size: 1.1rem; margin-top: 0.5rem;">Upload an audio file and click <strong>Analyze Audio</strong> to begin</div>
</div>
""", unsafe_allow_html=True)
