import os

# Base paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Dataset Location Logic
# We assume the dataset is in the sibling directory 'Image_classification/Dataset'
# or in a common 'Dataset' folder in the workspace root.
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
SIBLING_DATASET_DIR = os.path.join(WORKSPACE_ROOT, "Image_classification", "Dataset")

# Dataset Paths
UNIFIED_DIR = os.path.join(SIBLING_DATASET_DIR, "Unified_All_Sources")
NEU_DIR = os.path.join(SIBLING_DATASET_DIR, "NEU_Surface_Defects", "NEU-CLS")
SCRAPED_DIR = os.path.join(SIBLING_DATASET_DIR, "WebScraped_Products")
MVTEC_DIR = os.path.join(SIBLING_DATASET_DIR, "MVTec_Binary")

# Logic to pick active dataset
if os.path.exists(UNIFIED_DIR) and len(os.listdir(UNIFIED_DIR)) > 0:
    DATASET_DIR = UNIFIED_DIR
elif os.path.exists(MVTEC_DIR) and len(os.listdir(MVTEC_DIR)) > 0:
    DATASET_DIR = MVTEC_DIR
elif os.path.exists(SCRAPED_DIR) and len(os.listdir(SCRAPED_DIR)) > 0:
    DATASET_DIR = SCRAPED_DIR
elif os.path.exists(NEU_DIR):
    DATASET_DIR = NEU_DIR
else:
    # Fallback/Empty
    DATASET_DIR = os.path.join(PROJECT_ROOT, "data") 

MODELS_DIR = os.path.join(PROJECT_ROOT, "app", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Dynamic Class Detection
def get_classes(dataset_path):
    train_dir = os.path.join(dataset_path, "train")
    target_dir = train_dir if os.path.exists(train_dir) else dataset_path
    
    if not os.path.exists(target_dir):
        return []
        
    classes = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith('.')]
    classes.sort()
    return classes

CLASS_NAMES = get_classes(DATASET_DIR)
if not CLASS_NAMES:
    CLASS_NAMES = ["good", "defect"]

print(f"Using Dataset at: {DATASET_DIR}")
print(f"Classes: {CLASS_NAMES}")
