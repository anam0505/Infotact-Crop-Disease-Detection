import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Dynamic Path Resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_cnn.keras")

CLASS_NAMES = ["Tomato_Early_blight", "Tomato_healthy", "Tomato_Late_blight"]

def load_and_preprocess(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at path: {image_path}")
        
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    return np.expand_dims(img_array, axis=0)

def predict(model_path, image_path):
    if not os.path.exists(model_path):
        if os.path.exists(FALLBACK_MODEL_PATH):
            print(f"[WARN] Optimized model not found. Using baseline model from: {FALLBACK_MODEL_PATH}")
            model_path = FALLBACK_MODEL_PATH
        else:
            raise FileNotFoundError("❌ No trained model weights found in your 'models/' directory.")
            
    model = tf.keras.models.load_model(model_path)
    input_tensor = load_and_preprocess(image_path)
    
    predictions = model.predict(input_tensor, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx] * 100
    
    return CLASS_NAMES[pred_idx], confidence, predictions[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python src/predict.py <path_to_leaf_image>")
        print("Example: python src/predict.py data/val/Tomato_healthy/sample.jpg\n")
        sys.exit(1)
        
    test_img_path = sys.argv[1]
    
    try:
        label, conf, all_probs = predict(DEFAULT_MODEL_PATH, test_img_path)
        print("\n" + "="*50)
        print("           🌿 DIAGNOSIS RESULT REPORT")
        print("="*50)
        print(f" 🎯 Predicted Status : {label}")
        print(f" 📊 Confidence Score : {conf:.2f}%")
        print("-" * 50)
        print(" Class Probability Breakdown:")
        for idx, c_name in enumerate(CLASS_NAMES):
            print(f"  • {c_name:<22}: {all_probs[idx]*100:.2f}%")
        print("="*50 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}\n")