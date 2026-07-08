import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from augment import get_augmentation_pipeline

# 1. Configuration & Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_SAVE_PATH = "models/optimized_mobilenet.keras"

print(f"[INFO] Initializing Optimized Training on TensorFlow {tf.__version__}")

# 2. Dataset Loading (Using modern high-performance tf.data API)
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("❌ Ensure 'data/train' and 'data/val' directories exist with image folders.")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"[INFO] Detected {NUM_CLASSES} classes: {class_names}")

# Optimize dataset loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# 3. Build Transfer Learning Architecture
def build_model(num_classes):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    # Apply real-time augmentation (active during training only)
    x = get_augmentation_pipeline(IMG_SIZE)(inputs)
    
    # Preprocess specifically for MobileNetV2 scaling [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Load base model with pre-trained ImageNet weights
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze base layers initially
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, outputs, name="Optimized_Crop_Classifier")

model = build_model(NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Callbacks for Stability
os.makedirs("models", exist_ok=True)
callbacks_list = [
    callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1)
]

# 5. Execute Training
print("\n[INFO] Starting training loop...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks_list
)

print(f"\n✅ Training complete! Best model saved to: {MODEL_SAVE_PATH}")