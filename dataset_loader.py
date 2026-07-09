# dataset_loader.py
import os
from sklearn.model_selection import train_test_split

def get_data_splits(data_dir="PlantVillage/", test_size=0.2):
    """
    Scans the dataset directory, gathers all image paths and labels, 
    and splits them strictly into train and validation sets.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' not found. Please check your path.")

    all_paths = []
    all_labels = []
    
    # FIX: Strictly filter for valid directories only, ignoring hidden OS files like .DS_Store
    classes = sorted([
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"Detected valid classes ({len(classes)}): {class_to_idx}")
    
    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append(os.path.join(cls_path, img_name))
                all_labels.append(class_to_idx[cls_name])
                    
    # Strict stratified split to prevent class imbalance across train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    print(f"Data split complete: {len(train_paths)} training images, {len(val_paths)} validation images.")
    return train_paths, val_paths, train_labels, val_labels

if __name__ == "__main__":
    try:
        print("Testing dataset loader...")
        train_p, val_p, train_l, val_l = get_data_splits(data_dir="PlantVillage/")
        
        print("\n=== SUCCESS ===")
        print(f"Total training paths generated: {len(train_p)}")
        print(f"Total validation paths generated: {len(val_p)}")
        
    except Exception as e:
        print(f"\n=== ERROR ===\n{e}")