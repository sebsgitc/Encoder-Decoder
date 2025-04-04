import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import tifffile as tiff
from glob import glob
from sklearn.model_selection import train_test_split

# Define paths
PREPROCESSED_DIR = "3d-stacks/r04_/"  # Updated input path
SEGMENTATION_DIR = "output/segmentation_2d_stack/"
MODEL_DIR = "output/models/"
TARGET_SIZE = (256, 256)
os.makedirs(MODEL_DIR, exist_ok=True)

class LungSegmentationDataset(Dataset):
    def __init__(self, image_paths, vessel_masks, transform=None):
        self.image_paths = image_paths
        self.vessel_masks = vessel_masks
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = tiff.imread(self.image_paths[idx])
        vessel_mask = tiff.imread(self.vessel_masks[idx])
        
        # Resize to 1024x1024
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        vessel_mask = cv2.resize(vessel_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Scale to 8-bit and normalize to [0,1]
        image = (image / image.max() * 255).astype(np.uint8)
        image = image.astype(np.float32) / 255.0
        
        # Create binary vessel mask
        vessel_mask = (vessel_mask > 0).astype(np.int64)
        
        # Create 3-class mask
        mask = np.zeros_like(vessel_mask, dtype=np.int64)
        tissue_threshold = 30 / 255.0  # Normalized threshold
        mask[image > tissue_threshold] = 1  # Alveoli
        mask[vessel_mask > 0] = 2  # Blood vessels
        
        # Convert to tensors
        image = torch.FloatTensor(image).unsqueeze(0)
        mask = torch.LongTensor(mask)
        
        return image, mask

def prepare_data():
    """Prepare data with consistent file patterns"""
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocessed_dir = os.path.join(base_dir, "3d-stacks")  # Changed to look in root 3d-stacks folder
    segmentation_dir = os.path.join(base_dir, "output/segmentation_2d_stack")
    
    print(f"Looking for files in:")
    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Segmentation dir: {segmentation_dir}")

    # Look for files in all subfolders
    preprocessed_files = []
    segmentation_files = []

    # Walk through subfolders in preprocessed directory
    for subfolder in ["r04_"]:  # Add more subfolders if needed
        subfolder_path = os.path.join(preprocessed_dir, subfolder)
        if os.path.exists(subfolder_path):
            files = sorted(glob(os.path.join(subfolder_path, "*.tif")))
            preprocessed_files.extend(files)
            
            # Get corresponding segmentation files
            for f in files:
                seg_file = os.path.join(segmentation_dir, os.path.basename(f))
                if os.path.exists(seg_file):
                    segmentation_files.append(seg_file)

    print(f"Found {len(preprocessed_files)} preprocessed files")
    print(f"Found {len(segmentation_files)} segmentation files")

    if len(preprocessed_files) == 0 or len(segmentation_files) == 0:
        raise ValueError("No matching files found in preprocessed and segmentation directories")

    if len(preprocessed_files) != len(segmentation_files):
        raise ValueError(f"Mismatch in number of files: {len(preprocessed_files)} preprocessed vs {len(segmentation_files)} segmentation")

    # Split data with minimum validation size
    if len(preprocessed_files) < 5:
        val_size = 1
    else:
        val_size = 0.2

    train_images, val_images, train_vessels, val_vessels = train_test_split(
        preprocessed_files, segmentation_files, 
        test_size=val_size,
        random_state=42
    )
    
    return train_images, val_images, train_vessels, val_vessels