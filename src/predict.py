import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# =====================================================================
# VERIFIED ASCII ALPHABETICAL ORDERING
# Index 0: Tomato_Early_blight | Index 1: Tomato_Late_blight | Index 2: Tomato_healthy
# =====================================================================
CLASS_NAMES = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

def predict_image(interpreter, pil_image, class_names=CLASS_NAMES):
    """
    Shared inference logic using TensorFlow Lite Interpreter.
    Takes a PIL Image, converts/resizes it, and returns prediction metrics.
    """
    # 1. Get input/output tensor details from the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. Dynamically extract the target size the model expects (usually 224x224)
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (224, 224)
    
    # 3. Ensure RGB format and resize
    img_rgb = pil_image.convert("RGB")
    img_resized = img_rgb.resize(target_size)
    
    # 4. Convert to NumPy array as Float32 (No / 255.0 division if scaling is embedded!)
    img_array = np.array(img_resized, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # 5. Execute TFLite inference
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Safety Check: If the TFLite model outputs raw logits instead of Softmax probabilities, convert them.
    if np.sum(predictions) > 1.1 or np.sum(predictions) < 0.9:
        exp_preds = np.exp(predictions - np.max(predictions))
        predictions = exp_preds / exp_preds.sum()
    
    # 6. Extract winning label and confidence percentage
    pred_idx = np.argmax(predictions)
    confidence = float(predictions[pred_idx] * 100)
    predicted_label = class_names[pred_idx]
    
    # 7. Format all class probabilities as a dictionary for clean Streamlit UI charting
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
    
    # Resolve project root paths to find .tflite files
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(BASE_DIR, "models", "optimized_mobilenet.tflite")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", "baseline_cnn.tflite")
        
    if not os.path.exists(model_path):
        print("❌ [ERROR] No trained .tflite model weights found in 'models/' directory.")
        sys.exit(1)
        
    try:
        print(f"[INFO] Loading TFLite model from: {model_path}...")
        
        # Load TFLite Model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        if not os.path.exists(test_img_path):
            raise FileNotFoundError(f"Image not found at path: {test_img_path}")
            
        test_img = Image.open(test_img_path)
        label, conf, all_probs = predict_image(interpreter, test_img, CLASS_NAMES)
        
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
