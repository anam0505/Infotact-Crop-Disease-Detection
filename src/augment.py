import tensorflow as tf
from tensorflow.keras import layers, models

def get_augmentation_pipeline(img_size=(224, 224)):
    """
    Returns a Keras Sequential model that applies real-time 
    spatial and color augmentations to input images.
    """
    augmentation_pipeline = models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
    ], name="leaf_data_augmentation")
    
    return augmentation_pipeline

if __name__ == "__main__":
    # Test pipeline initialization
    pipeline = get_augmentation_pipeline()
    print("✅ Augmentation pipeline initialized successfully:")
    pipeline.summary()