import os
import cv2
import numpy as np
from app import config

def load_from_directory(directory_path):
    """
    Load images and labels.
    """
    images = []
    labels = []
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return [], []
        
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(directory_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Loading {class_name} from {class_dir}...")
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(class_dir, filename)
                img = cv2.imread(file_path) # Read as Color (BGR) for DETR
                if img is not None:
                    images.append(img)
                    labels.append(label_idx)
    
    return images, np.array(labels)

def load_data():
    train_dir = os.path.join(config.DATASET_DIR, 'train')
    val_dir = os.path.join(config.DATASET_DIR, 'val')
    
    # If no train/val split, just load from root (simple case)
    if not os.path.exists(train_dir):
        print(f"No train/val split found in {config.DATASET_DIR}. Loading from root...")
        return load_from_directory(config.DATASET_DIR), ([], [])

    print("Loading Training Data...")
    train_X, train_y = load_from_directory(train_dir)
    
    print("Loading Validation Data...")
    val_X, val_y = load_from_directory(val_dir)
    
    return (train_X, train_y), (val_X, val_y)
