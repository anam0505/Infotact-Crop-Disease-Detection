# src/realtime_camera.py
import os
import time
import cv2
import numpy as np
import tensorflow as tf

# Resolve project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_cnn.keras")

# VERIFIED ASCII ALPHABETICAL ORDERING
CLASS_NAMES = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

def load_live_model():
    if os.path.exists(DEFAULT_MODEL_PATH):
        print(f"[INFO] Loading Optimized MobileNetV2 from: {DEFAULT_MODEL_PATH}")
        return tf.keras.models.load_model(DEFAULT_MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        print(f"[WARN] Optimized model missing. Loading Baseline CNN from: {FALLBACK_MODEL_PATH}")
        return tf.keras.models.load_model(FALLBACK_MODEL_PATH)
    else:
        raise FileNotFoundError("❌ No trained model found in 'models/' directory.")

def run_realtime_stream(camera_index=0):
    model = load_live_model()
    
    print(f"[INFO] Initializing Video Stream (Camera Index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open video device. Check webcam connections or camera permissions.")
        
    print("\n" + "="*55)
    print(" 🌿 REAL-TIME CROP DIAGNOSTIC STREAM ACTIVE")
    print(" 💡 Press the 'q' key on your keyboard to exit the stream.")
    print("="*55 + "\n")
    
    fps_start_time = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera. Exiting...")
            break
            
        frame = cv2.flip(frame, 1) # Mirror frame
        
        # Calculate Real-Time FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        fps_start_time = fps_end_time
        
        # Preprocess Frame for CNN Inference (Do NOT divide by 255.0!)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        input_tensor = np.expand_dims(np.array(resized_frame, dtype=np.float32), axis=0)
        
        # Execute Prediction
        predictions = model.predict(input_tensor, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx] * 100)
        label = CLASS_NAMES[pred_idx]
        
        # Dynamic Visual Overlays
        if "Healthy" in label:
            color = (0, 255, 0)      # BGR Green
            status_icon = "STATUS: SAFE"
        else:
            color = (0, 0, 255)      # BGR Red
            status_icon = "ALERT: PATHOGEN DETECTED"
            
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), color, 2)
        
        cv2.putText(frame, f"DIAGNOSIS: {label} ({confidence:.1f}%)", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"{status_icon} | System FPS: {fps:.1f}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    
        cv2.imshow("Real-Time Crop Disease Vision Engine", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Stream terminated by user.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_realtime_stream()
    except Exception as e:
        print(f"\n[ERROR] Stream failed: {e}\n")