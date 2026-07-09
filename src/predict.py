# src/predict.py
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# =====================================================================
# VERIFIED ASCII ALPHABETICAL ORDERING (Matches your terminal output exactly)
# Index 0: Tomato_Early_blight | Index 1: Tomato_Late_blight | Index 2: Tomato_healthy
# =====================================================================
CLASS_NAMES = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

def predict_image(model, pil_image, class_names=CLASS_NAMES, target_size=(224, 224)):
    """
    Shared inference logic used by both app.py and command-line scripts.
    Takes a PIL Image, converts/resizes it, and returns prediction metrics.
    """
    # 1. Ensure RGB format and resize to match neural network input shape (224x224)
    img_rgb = pil_image.convert("RGB")
    img_resized = img_rgb.resize(target_size)
    
    # 2. Convert to NumPy array as Float32
    # CRITICAL FIX: Do NOT divide by 255.0! The saved Keras model already contains
    # tf.keras.applications.mobilenet_v2.preprocess_input(), which expects raw [0, 255] pixels.
    img_array = np.array(img_resized, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # 3. Execute inference
    predictions = model.predict(img_tensor, verbose=0)[0]
    
    # 4. Extract winning label and confidence percentage
    pred_idx = np.argmax(predictions)
    confidence = float(predictions[pred_idx] * 100)
    predicted_label = class_names[pred_idx]
    
    # 5. Format all class probabilities as a dictionary for clean Streamlit UI charting
    all_probs_dict = {
        class_names[i]: float(predictions[i] * 100) for i in range(len(class_names))
    }
    
    return predicted_label, confidence, all_probs_dict

# =====================================================================
# COMMAND-LINE EXECUTION LOGIC (For standalone terminal testing)
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python src/predict.py <path_to_leaf_image>")
        print("Example: python src/predict.py data/val/Tomato_healthy/sample.jpg\n")
        sys.exit(1)
        
    test_img_path = sys.argv[1]
    
    # Resolve project root paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", "baseline_cnn.keras")
        
    if not os.path.exists(model_path):
        print("❌ [ERROR] No trained model weights found in 'models/' directory.")
        sys.exit(1)
        
    try:
        print(f"[INFO] Loading model from: {model_path}...")
        loaded_model = tf.keras.models.load_model(model_path)
        
        if not os.path.exists(test_img_path):
            raise FileNotFoundError(f"Image not found at path: {test_img_path}")
            
        test_img = Image.open(test_img_path)
        label, conf, all_probs = predict_image(loaded_model, test_img, CLASS_NAMES)
        
        print("\n" + "="*50)
        print("           🌿 DIAGNOSIS RESULT REPORT")
        print("="*50)
        print(f" 🎯 Predicted Status : {label}")
        print(f" 📊 Confidence Score : {conf:.2f}%")
        print("-" * 50)
        print(" Class Probability Breakdown:")
        for c_name, c_prob in all_probs.items():
            print(f"  • {c_name:<22}: {c_prob:.2f}%")
        print("="*50 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}\n")