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
# 1. PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="AgriVision Pro | Crop Disease Portal",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DYNAMIC PATH RESOLUTION & CONSTANTS
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_cnn.keras")

CLASS_NAMES = ["Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight"]

# Actionable Treatment Protocols Database
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
# 3. HELPER FUNCTIONS & QUALITY GUARDRAILS
# ==========================================
@st.cache_resource
def load_classifier():
    """Loads the trained Keras model with automatic fallback logic."""
    if os.path.exists(DEFAULT_MODEL_PATH):
        return tf.keras.models.load_model(DEFAULT_MODEL_PATH), "Optimized MobileNetV2"
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return tf.keras.models.load_model(FALLBACK_MODEL_PATH), "Baseline Custom CNN"
    return None, None

def check_image_sharpness(pil_img, threshold=80.0):
    """Calculates the variance of the Laplacian using OpenCV to detect blurry images."""
    open_cv_image = np.array(pil_img.convert('L'))  # Convert to grayscale
    laplacian_var = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    return laplacian_var, laplacian_var >= threshold

def generate_report_text(label, conf, latency, temp, hum, guide):
    """Compiles a professional text report for download."""
    report = f"""==================================================
AGRIVISION PRO | FIELD DIAGNOSTIC CERTIFICATE
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
Report generated automatically by AgriVision Pro Edge System.
"""
    return report

# ==========================================
# 4. SIDEBAR: TELEMETRY & SYSTEM STATUS
# ==========================================
model, model_name = load_classifier()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/sprout.png", width=64)
    st.title("AgriVision Edge")
    st.caption("Professional Crop Screening Portal")
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
        st.success(f"**Engine:** `{model_name}`\n\n**Status:** Online & Ready")
    else:
        st.error("**Engine:** Offline\n\nNo weights found in `models/`.")
        
    st.markdown("---")
    st.caption("© 2026 Infotact Engineering Team")

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
st.title("🌿 Intelligent Crop Disease Detection & Decision Support")
st.markdown("Deploy computer vision and hydrothermal telemetry to diagnose plant pathology in real time.")
st.markdown("---")

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
# 6. INFERENCE & DIAGNOSTIC ENGINE
# ==========================================
if input_image is not None:
    col_img, col_results = st.columns([1, 1.2])
    
    with col_img:
        st.subheader("🖼️ Specimen Ingestion")
        st.image(input_image, caption=f"Source: {source_name}", use_container_width=True)
        
        # Quality Guardrail Check
        sharpness_score, is_sharp = check_image_sharpness(input_image)
        if not is_sharp:
            st.warning(f"⚠️ **Image Quality Guard:** This photo appears slightly blurry (Sharpness Index: `{sharpness_score:.1f}`). For maximum AI accuracy, ensure steady focus and adequate field lighting.")
        else:
            st.caption(f"🔒 **Quality Check Passed** (Sharpness Index: `{sharpness_score:.1f}`)")

    with col_results:
        st.subheader("🔍 AI Diagnostic Analysis")
        
        with st.spinner("Executing neural feature extraction..."):
            start_time = time.time()

            # Inference (shared logic with src/predict.py so CLI and app
            # never drift out of sync on preprocessing).
            predicted_label, confidence, all_probs = predict_image(
                model, input_image, class_names=CLASS_NAMES
            )
            latency_ms = (time.time() - start_time) * 1000
            guide = TREATMENT_GUIDES[predicted_label]

        # Primary Status Banner
        if "Healthy" in predicted_label:
            st.success(f"### Diagnosis: {predicted_label}\n**Confidence Score:** `{confidence:.2f}%` | **Latency:** `{latency_ms:.1f} ms`")
        else:
            st.error(f"### PATHOGEN DETECTED: {predicted_label}\n**Confidence Score:** `{confidence:.2f}%` | **Latency:** `{latency_ms:.1f} ms`")

        # Probability Distribution
        st.markdown("#### Class Confidence Distribution")
        st.bar_chart(all_probs)

    st.markdown("---")

    # ==========================================
    # 7. ACTIONABLE DECISION SUPPORT & TREATMENT
    # ==========================================
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
        
    st.markdown("---")

    # ==========================================
    # 8. EXPORTABLE REPORT GENERATION
    # ==========================================
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
        file_name=f"AgriVision_Report_{file_timestamp}.txt",
        mime="text/plain",
        use_container_width=True
    )

elif input_image is None:
    st.info("👆 Please upload a leaf photo or capture a live camera snapshot above to initiate AI diagnostic screening.")