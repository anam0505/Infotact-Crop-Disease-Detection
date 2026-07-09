import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

# Dynamic Path Resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Please run train_optimized.py first!")
        
    print(f"[INFO] Loading trained model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )
    
    class_names = val_dataset.class_names
    n_classes = len(class_names)
    
    # Extract labels and predict
    print("[INFO] Generating predictions on validation set...")
    y_true, y_pred_probs = [], []
    
    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Print Metrics Table
    print("\n" + "="*55)
    print("               CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Crop Disease Detection")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {cm_path}")
    
    # Save Multi-Class ROC Curves
    y_test_bin = label_binarize(y_true_classes, classes=range(n_classes))
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[INFO] ROC curves saved to: {roc_path}")

if __name__ == "__main__":
    evaluate_model()