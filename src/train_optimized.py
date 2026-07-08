# src/train_optimized.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from src.augment import get_augmentation_pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "optimized_mobilenet.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

print(f"[INFO] Initializing Training on TensorFlow {tf.__version__}")

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

print(f"[INFO] Detected Classes ({len(train_dataset.class_names)}): {train_dataset.class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

def build_model(num_classes):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    # 1. Apply spatial data augmentations
    x = get_augmentation_pipeline(IMG_SIZE)(inputs)
    
    # 2. Preprocess specifically for MobileNetV2 [-1, 1] scaling
    # This automatically takes raw integer RGB values [0, 255] and converts them!
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # 3. Load pre-trained ImageNet weights
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, outputs, name="Optimized_Crop_Classifier")

model = build_model(len(train_dataset.class_names))
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
callbacks_list = [
    callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks_list
)

print(f"\n✅ Optimization complete! Best model weights saved to: {MODEL_SAVE_PATH}")