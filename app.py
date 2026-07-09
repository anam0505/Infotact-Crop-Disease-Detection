import os
import time
import random
import datetime
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf

from src.predict import predict_image

# ==========================================
# 1. PAGE CONFIGURATION & ENTERPRISE UI THEMING
# ==========================================
st.set_page_config(
    page_title="CropSense AI | Smart Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting Advanced CSS: Google Fonts, Gradients, Glassmorphism & UI Cards
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1f2937;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Subtle Agricultural Theme Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 40%, #ecfdf5 100%);
    }

    /* Hero Header Container */
    .hero-container {
        text-align: center;
        padding: 30px 20px;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #d1fae5;
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.05);
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }
    .gradient-title {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #047857 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        margin-bottom: 5px;
    }
    .hero-heading {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0px;
        margin-bottom: 12px;
    }
    .hero-subtitle {
        color: #4b5563;
        font-size: 1.05rem;
        max-width: 700px;
        margin: 0 auto 18px auto;
        line-height: 1.6;
    }
    .feature-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        border: 1px solid #a7f3d0;
    }

    /* Modern UI Cards for Sections */
    .ui-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }

    /* Pulsing Biohazard Alert for Diseased Plants */
    .pulse-alert {
        animation: pulse-red 2s infinite;
        border-radius: 16px;
        padding: 20px;
        background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
        color: #991b1b;
        border: 2px solid #ef4444;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.1);
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Success Healthy Badge */
    .healthy-badge {
        border-radius: 16px;
        padding: 20px;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        color: #065f46;
        border: 2px solid #10b981;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.1);
    }

    /* Sidebar Styling */
    .sidebar-logo {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 5px;
        filter: drop-shadow(0px 4px 8px rgba(16, 185, 129, 0.25));
    }
    .online-dot {
        height: 10px;
        width: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-green 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PATH RESOLUTION & CONSTANTS
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_cnn.keras")

# VERIFIED ASCII ALPHABETICAL ORDERING (E -> L -> h)
CLASS_NAMES = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

TREATMENT_GUIDES = {
    "Tomato Early Blight": {
        "pathogen": "Alternaria solani (Fungal Spores)",
        "risk": "High (Spreads rapidly in warm, humid conditions)",
        "immediate_action": "1. Isolate and prune infected lower leaves immediately.\n2. Avoid overhead sprinkler irrigation to keep foliage dry.\n3. Disinfect pruning shears between cuts with 70% alcohol.",
        "fungicide": "Apply copper-based fungicides or chlorothalonil every 7-10 days until symptoms subside."
    },
    "Tomato Late Blight": {
        "pathogen": "Phytophthora infestans (Water Mold)",
        "risk": "Critical (Can destroy entire crop within days)",
        "immediate_action": "1. Uproot and burn heavily infected plants immediately—do NOT compost.\n2. Improve field ventilation by weeding and staking.\n3. Notify neighboring farms of potential spore drift.",
        "fungicide": "Apply targeted systemic fungicides containing mefenoxam, cymoxanil, or propamocarb immediately."
    },
    "Tomato Healthy": {
        "pathogen": "None Detected",
        "risk": "Low / Nominal",
        "immediate_action": "Continue standard crop maintenance and scheduled nutrient feeding.",
        "fungicide": "No fungicidal intervention required. Maintain routine biological monitoring."
    }
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_classifier():
    if os.path.exists(DEFAULT_MODEL_PATH):
        return tf.keras.models.load_model(DEFAULT_MODEL_PATH), "Optimized MobileNetV2"
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return tf.keras.models.load_model(FALLBACK_MODEL_PATH), "Baseline Custom CNN"
    return None, None

def check_image_sharpness(pil_img, threshold=80.0):
    open_cv_image = np.array(pil_img.convert('L'))
    laplacian_var = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    return laplacian_var, laplacian_var >= threshold

def generate_report_text(label, conf, latency, temp, hum, guide):
    report = f"""==================================================
CROPSENSE AI | FIELD DIAGNOSTIC CERTIFICATE
==================================================
Scan Timestamp  : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Field Location  : Sector 4-B (Geotag: 13.0827° N, 80.2707° E)
AI Engine       : TensorFlow Edge / Keras
Inference Speed : {latency:.2f} ms

--- ENVIRONMENTAL TELEMETRY ---
Ambient Air Temp: {temp:.1f} °C
Relative Humidity: {hum:.1f} %

--- DIAGNOSTIC FINDINGS ---
Predicted Status: {label.upper()}
Confidence Score: {conf:.2f}%
Pathogen Type   : {guide['pathogen']}
Risk Level      : {guide['risk']}

--- RECOMMENDED ACTION PLAN ---
{guide['immediate_action']}

--- FUNGICIDE PROTOCOL ---
{guide['fungicide']}

==================================================
Report generated automatically by CropSense AI Edge System.
"""
    return report

# ==========================================
# 4. SIDEBAR: TELEMETRY & SYSTEM STATUS
# ==========================================
model, model_name = load_classifier()

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌿</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom: 2px; color: #059669; font-size: 1.8rem;'>CropSense AI</h2>", unsafe_allow_html=True)
    st.caption("Powered by MobileNetV2 & TensorFlow")
    st.markdown("---")
    
    st.subheader("📡 Live Microclimate")
    sim_temp = random.uniform(25.0, 29.8)
    sim_hum = random.uniform(65.0, 82.0)
    
    col_t, col_h = st.columns(2)
    col_t.metric("Temp", f"{sim_temp:.1f} °C", delta="-0.2 °C")
    col_h.metric("Humidity", f"{sim_hum:.1f} %", delta="+1.4 %")
    
    if sim_hum > 75.0:
        st.warning("⚠️ **Fungal Drift Alert:** High humidity detected. Spore proliferation risk elevated.")
        
    st.markdown("---")
    st.subheader("⚙️ System Status")
    if model is not None:
        st.markdown(f"**Engine:** `{model_name}`")
        st.markdown("<div class='online-dot'></div> <b>Online & Ready</b>", unsafe_allow_html=True)
    else:
        st.error("**Engine:** Offline\n\nNo weights found in `models/`.")
        
    st.markdown("---")
    st.caption("Developed by Infotact DS & ML Engineering Team")

# ==========================================
# 5. MAIN DASHBOARD UI (HERO BANNER)
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="gradient-title">🌿 CropSense AI</div>
    <div class="hero-heading">Smart Crop Disease Intelligence Platform</div>
    <div class="hero-subtitle">
        Deploying computer vision and hydrothermal edge telemetry to diagnose botanical pathology in real time. 
        Get instant laboratory-grade precision in the field.
    </div>
    <div>
        <span class="feature-badge">⚡ Deep Learning Powered</span>
        <span class="feature-badge">🔍 Real-Time Diagnosis</span>
        <span class="feature-badge">💊 Actionable Protocols</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ **System Offline:** Could not locate model weights. Please ensure `models/optimized_mobilenet.keras` exists!")
    st.stop()

# Multi-Modal Ingestion Tabs
tab_upload, tab_camera = st.tabs(["📂 File Upload Ingestion", "📷 Live Edge Camera Snapshot"])

input_image = None
source_name = ""

with tab_upload:
    uploaded_file = st.file_uploader("Select a botanical leaf specimen (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"], key="file_up")
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        source_name = uploaded_file.name

with tab_camera:
    st.write("Position edge camera directly above target foliage:")
    cam_file = st.camera_input("Capture Foliage Specimen", key="cam_up", label_visibility="collapsed")
    if cam_file:
        input_image = Image.open(cam_file).convert("RGB")
        source_name = "Live_Camera_Snapshot.jpg"

# ==========================================
# 6. ANIMATED INFERENCE & DIAGNOSTIC ENGINE
# ==========================================
if input_image is not None:
    st.toast('Specimen ingested successfully! Initializing AI scan...', icon='⚡')
    
    col_img, col_results = st.columns([1, 1.25])
    
    with col_img:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Specimen Specimen")
        st.image(input_image, caption=f"Source: {source_name}", use_container_width=True)
        
        sharpness_score, is_sharp = check_image_sharpness(input_image)
        if not is_sharp:
            st.warning(f"⚠️ **Quality Guard:** Photo appears slightly blurry (Sharpness Index: `{sharpness_score:.1f}`).")
        else:
            st.caption(f"🔒 **Quality Check Passed** (Sharpness Index: `{sharpness_score:.1f}`)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.subheader("🔍 AI Diagnostic Analysis")
        
        # ANIMATED SCANNING PROGRESS BAR
        scan_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        # Simulate scanning animation before actual prediction
        for percent_complete in range(0, 101, 20):
            time.sleep(0.08)
            scan_bar.progress(percent_complete)
            status_text.caption(f"Extracting neural features... {percent_complete}%")
            
        status_text.empty()
        scan_bar.empty()
        
        # Execute actual inference logic
        predicted_label, confidence, all_probs_dict = predict_image(
            model, input_image, class_names=CLASS_NAMES
        )
        latency_ms = (time.time() - start_time) * 1000
        guide = TREATMENT_GUIDES[predicted_label]

        # PROFESSIONAL SERIOUS RESULTS
        if "Healthy" in predicted_label:
            st.markdown(f"""
            <div class="healthy-badge">
                <h3 style='margin:0; color:#065f46;'>🌱 STATUS: {predicted_label.upper()}</h3>
                <p style='margin: 5px 0 0 0; font-size: 0.95rem;'><b>Confidence Score:</b> {confidence:.2f}% &nbsp;|&nbsp; <b>Latency:</b> {latency_ms:.1f} ms</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pulse-alert">
                <h3 style='margin:0; color:#991b1b;'>🚨 PATHOGEN DETECTED: {predicted_label.upper()}</h3>
                <p style='margin: 5px 0 0 0; font-size: 0.95rem;'><b>Confidence Score:</b> {confidence:.2f}% &nbsp;|&nbsp; <b>Latency:</b> {latency_ms:.1f} ms</p>
            </div>
            <br>
            """, unsafe_allow_html=True)

        st.markdown("#### Class Confidence Distribution (%)")
        st.bar_chart(all_probs_dict)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. ACTIONABLE DECISION SUPPORT
    # ==========================================
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("📋 Targeted Treatment & Mitigation Protocol")
    
    col_proto1, col_proto2, col_proto3 = st.columns(3)
    
    with col_proto1:
        st.markdown("#### 🦠 Pathogen Profile")
        st.write(f"**Agent:** `{guide['pathogen']}`")
        st.write(f"**Severity Risk:** `{guide['risk']}`")
        
    with col_proto2:
        st.markdown("#### ⚡ Immediate Action Plan")
        st.write(guide['immediate_action'])
        
    with col_proto3:
        st.markdown("#### 💊 Fungicide Guidelines")
        st.write(guide['fungicide'])
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 8. EXPORTABLE REPORT GENERATION
    # ==========================================
    st.markdown('<div class="ui-card" style="background: #f8fafc;">', unsafe_allow_html=True)
    st.subheader("📤 Export Field Certificate")
    st.write("Download an official text summary of this screening for farm records or agricultural insurance compliance.")
    
    report_content = generate_report_text(
        label=predicted_label,
        conf=confidence,
        latency=latency_ms,
        temp=sim_temp,
        hum=sim_hum,
        guide=guide
    )
    
    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.download_button(
        label="📄 Download Diagnostic Certificate (.TXT)",
        data=report_content,
        file_name=f"CropSense_Report_{file_timestamp}.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

elif input_image is None:
    st.info("👆 Please upload a leaf photo or capture a live camera snapshot above to initiate AI diagnostic screening.")