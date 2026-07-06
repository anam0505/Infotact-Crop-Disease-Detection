# dataset_loader.py
import os
from sklearn.model_selection import train_test_split# Make sure scikit-learn is installed

def get_data_splits(data_dir="PlantVillage/", test_size=0.2):
    """
    Scans the dataset directory, gathers all image paths and labels, 
    and splits them strictly into train and validation sets.
    """
    all_paths = []
    all_labels = []
    
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        if os.path.isdir(cls_path):
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_paths.append(os.path.join(cls_path, img_name))
                    all_labels.append(class_to_idx[cls_name])
                    
    # Strict chronological/random split to prevent data leakage
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    print(f"Data split complete: {len(train_paths)} training images, {len(val_paths)} validation images.")
    return train_paths, val_paths, train_labels, val_labels

# Make sure this has NO indentation (starts at the absolute left margin)
if __name__ == "__main__":
    try:
        print("Testing dataset loader...")
        train_p, val_p, train_l, val_l = get_data_splits(data_dir="PlantVillage/")
        
        print("\n=== SUCCESS ===")
        print(f"Total training paths generated: {len(train_p)}")
        print(f"Total validation paths generated: {len(val_p)}")
        
    except Exception as e:
        print(f"\n=== ERROR ===\n{e}")