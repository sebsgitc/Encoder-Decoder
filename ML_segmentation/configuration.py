"""Configuration parameters for the lung segmentation project."""

import os

# Base directory
BASE_DIR = "/home2/ad4631sv-s/TestSebbe/Encoder-Decoder"

# Data parameters
DATA_DIR = os.path.join(BASE_DIR, "3d-stacks")  # Directory containing the original data
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # Directory containing segmentation results
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed")

# Sample data - expanded to include all specified image-mask pairs
SAMPLE_DATASETS = [
    {
        "name": "r01_", 
        "raw_file": "r01_/r01_.8bit.tif",
        "mask_file": "vessel_segmentation_r01_*.tif"  # Using wildcard to find the latest
    },
    {
        "name": "rL1_", 
        "raw_file": "rL1_/rL1_.8bit.tif",
        "mask_file": "vessel_segmentation_rL1_*.tif"
    },
    {
        "name": "r33_", 
        "raw_file": "r33_/r33_.8bit.tif",
        "mask_file": "vessel_segmentation_r33_*.tif"
    },
    {
        "name": "r34_", 
        "raw_file": "r34_/r34_.8bit.tif",
        "mask_file": "vessel_segmentation_r34_*.tif"
    },
    {
        "name": "rL4_", 
        "raw_file": "rL4_/rL4_.8bit.tif",
        "mask_file": "vessel_segmentation_rL4_*.tif"
    }#,
    # {
    #     "name": "r04_", 
    #     "raw_file": "r04_/r04_.8bit.tif",
    #     "mask_file": "vessel_segmentation_r04_*.tif"
    # }
]

# Model parameters
IMAGE_SIZE = 1024  # Resize dimensions
PATCH_SIZE = 256  # For patch extraction if needed
BATCH_SIZE = 4  # Set batch size to 4 instead of 2 as requested
NUM_WORKERS = 2  # Maintain 2 workers to avoid memory pressure
NUM_CLASSES = 2  # Background/other, blood vessels
FILTERS = [4, 8, 16, 32, 64]  # Increase filter sizes to use more GPU memory while maintaining performance
LEARNING_RATE = 1e-4  # Slightly higher learning rate for faster convergence
EPOCHS = 15  # Reduced epochs for faster testing
EARLY_STOPPING_PATIENCE = 5  # Increase early stopping patience for more stable training

# Training parameters
VAL_SPLIT = 0.2  # Fix validation split to 20% (meaning 80% training data)
RANDOM_SEED = 42
USE_PATCHES = True  # Whether to use patch-based training
USE_2D = True  # Whether to use 2D (True) or 3D (False) approach

# Multi-dimensional training settings
USE_MULTI_AXIS_SLICING = False  # Disable multi-axis slicing to ensure stability
MULTI_AXIS_BALANCE = [1.0, 0.0, 0.0]  # Only use Z-axis slices (100% Z, 0% Y, 0% X)
TRAIN_WITH_ORTHOGONAL_VIEWS = False  # Disable orthogonal views
USE_ALL_SLICES = True  # Use all Z-axis slices

# Define the sampling sizes for each axis (not used since we're only using Z-axis)
MULTI_AXIS_SAMPLE_SIZE = {
    'Z': 10000,  # Very large number to ensure all Z slices are used
    'Y': 0,      # No Y slices
    'X': 0       # No X slices
}

# Class weights for weighted loss - rebalance to prioritize vessel over background
CLASS_WEIGHTS = [0.25, 0.75]  # Significantly favor vessels over background

# Add vessel priority settings
VESSEL_PRIORITY = {
    'factor': 4.0,                  # Higher weight for vessel vs background importance
    'recall_target': 0.85,          # Target recall value for vessels
    'preservation_threshold': 0.75  # Minimum ratio of predicted to true vessel pixels
}

# Focal loss parameters - adjust to focus more on vessels
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.5  # Slightly increased from 2.0 to better focus on hard examples

# Add background-focused training parameters - updated to prioritize vessels over background
BACKGROUND_FOCUS = {
    'enable_hard_negative_mining': True,  # Focus on difficult background regions
    'hard_negative_ratio': 0.2,           # Reduced from 0.3 to focus more on vessels
    'background_boost_factor': 0.9,       # Reduced from 1.25 to reduce background emphasis
    'border_weight_factor': 2.0           # Increased from 1.5 to focus on vessel boundaries
}

# Add boundary-aware training options
BOUNDARY_AWARE_TRAINING = {
    'enable': False,               # Disable boundary-aware components until fixed
    'boundary_width': 3,          # Width of boundary region in pixels
    'boundary_weight': 2.0        # Weight for boundary regions
}

# Slice range to use (to limit memory usage when processing large volumes)
SLICE_RANGE = None  # Set to None to use all available slices, or specify a range like (150, 200)

# Debug settings
DEBUG_DATA_LOADING = False   # Disable detailed data loading information
DEBUG_DATA_AUGMENTATION = False  # Disable data augmentation debugging

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING = False  # Disable performance monitoring
SAVE_EVALUATION_IMAGES = True  # Save evaluation images for later viewing
EVALUATION_OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")  # Directory to save evaluation images

# Memory efficiency settings
MAX_MEMORY_USAGE_GB = 14  # Further reduce memory limit
PRELOAD_VOLUMES = False  # Don't preload entire volumes to save memory
PREPROCESS_BATCH_SIZE = 4  # Increase from 1 to 2

# Even smaller filters for the model to reduce memory footprint
DEFAULT_FILTERS = [4, 8, 16, 32, 64]

# Add mixed precision settings
USE_MIXED_PRECISION = False  # Temporarily disable mixed precision until type issues are fixed
MIXED_PRECISION_DTYPE = 'float32'  # Use float32 for now to avoid mixed precision issues

# If you still want to try mixed precision, use these settings instead:
# USE_MIXED_PRECISION = True  
# MIXED_PRECISION_DTYPE = 'mixed_float16'  # Use float16 for compute but keep variables in float32

# Add NaN handling settings
NAN_HANDLING = {
    'detect_early': True,          # Detect NaNs early in the pipeline
    'replace_nan_inputs': True,    # Replace NaN inputs with zeros
    'nan_safe_losses': True,       # Use NaN-safe loss calculations
    'monitor_nan_frequency': True  # Log frequency of NaN occurrences
}

# Create necessary directories
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)