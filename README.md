# 🌿 CropSense AI

**Smart Plant Health Intelligence Platform**

*Deep Learning Powered Disease Detection • Instant Diagnosis • Treatment Recommendations*

## 📖 Overview

**CropSense AI** is an enterprise-grade, computer vision web application designed to diagnose plant pathology in real-time. By leveraging a highly optimized MobileNetV2 deep learning architecture and simulating live hydrothermal edge telemetry, this platform provides farmers and agronomists with laboratory-grade precision in the field.

**Developed by:** Infotact DS & ML Engineering Team

## ✨ Key Features

* **Multi-Modal Ingestion:** Analyze crops via static image upload (JPG/PNG) or live edge camera snapshot.

* **Real-Time AI Diagnostics:** Sub-second inference powered by a transfer-learned MobileNetV2 neural network.

* **Automated Image Quality Guard:** Built-in OpenCV Laplacian variance checking to reject blurry or out-of-focus images, preserving model accuracy.

* **Environmental Telemetry Integration:** Simulates live microclimate data (Temperature & Humidity) to trigger biological fungal drift alerts.

* **Actionable Decision Support:** Generates structured pathogen profiles, immediate containment actions, and targeted fungicide protocols.

* **Exportable Diagnostics:** One-click generation of professional `.TXT` Field Diagnostic Certificates for farm records and compliance.

* **Enterprise UI/UX:** Features a custom CSS glassmorphism interface, animated neural scanning progress bars, and dynamic status badges.

## 📂 Project Structure

```text
.
├── app.py                        # Main Streamlit web dashboard (Frontend)
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git tracking rules
├── src/
│   ├── dataset_loader.py         # Advanced data splitting and loading logic
│   ├── train_optimized.py        # MobileNetV2 transfer learning & training script
│   ├── predict.py                # Shared inference module (Used by app.py and CLI)
│   ├── realtime_camera.py        # OpenCV live video feed inference script
│   └── augment.py                # Spatial data augmentation pipeline
├── models/                       
│   ├── optimized_mobilenet.keras # Primary production weights (Ignored in Git)
│   └── baseline_cnn.keras        # Fallback custom CNN weights
└── PlantVillage/                 # Root directory for training datasets