import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from unet_model import UNet
import cv2

# Define paths
PREPROCESSED_DIR = "3d-stacks/"
VESSEL_MASK_DIR = "output/segmentation_2d_stack/"
MODEL_DIR = "output/models/"
OUTPUT_DIR = "output/visualisation/"

# Add constant for image size
TARGET_SIZE = (256, 256)

def create_custom_colormap():
    """Create a custom colormap with distinguishable colours."""
    colours = [
        (0, 0, 0),          # Background (black)
        (1, 0.84, 0),       # Alveoli (gold)
        (0.86, 0.08, 0.24)  # Blood vessels (crimson)
    ]
    return LinearSegmentedColormap.from_list("custom", colours)

def load_model(model_path):
    """Load trained U-Net model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    # Load checkpoint with weights_only=True for security
    checkpoint = torch.load(
        model_path, 
        map_location=device,
        weights_only=True  # Added parameter to address warning
    )
    
    # Load state dict directly if using weights_only=True
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, device

def preprocess_image(image):
    """Preprocess image to 8-bit and resize to target size."""
    # Convert to 8-bit
    image_8bit = ((image - image.min()) * 255 / (image.max() - image.min())).astype(np.uint8)
    # Resize using OpenCV
    image_resized = cv2.resize(image_8bit, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return image_resized

def show_segmentation(original_filename):
    """Display original image, vessel mask, and U-Net segmentation."""
    print("Starting visualization process...")
    
    # Construct paths
    print("1/5: Setting up paths...")
    original_path = os.path.join(PREPROCESSED_DIR, original_filename)
    vessel_mask_path = os.path.join(VESSEL_MASK_DIR, original_filename)
    
    # Load and preprocess images (~1 second)
    print("2/5: Loading and preprocessing images...")
    original = tiff.imread(original_path)
    vessel_mask = tiff.imread(vessel_mask_path)
    
    # Preprocess images
    original_processed = preprocess_image(original)
    vessel_mask_processed = cv2.resize(vessel_mask, TARGET_SIZE, 
                                     interpolation=cv2.INTER_NEAREST)
    
    # Load latest model (~2-3 seconds on M1)
    print("3/5: Loading model...")
    model_files = sorted(os.listdir(MODEL_DIR))
    if not model_files:
        raise ValueError("No trained models found!")
    model_path = os.path.join(MODEL_DIR, model_files[-1])
    model, device = load_model(model_path)
    print(f"Using device: {device}")
    
    # Prepare input for U-Net (~1 second)
    print("4/5: Processing with U-Net...")
    input_tensor = torch.FloatTensor(original_processed).unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    # Get U-Net prediction (~2-3 seconds)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.argmax(prediction, dim=1).cpu().numpy()[0]
    
    # Rest of visualization code remains the same but use processed images
    print("5/5: Creating visualization...")
    custom_cmap = create_custom_colormap()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_processed, cmap="gray")
    axes[0].set_title("Original Image (256x256)")
    axes[0].axis("off")
    
    # Vessel mask
    axes[1].imshow(original_processed, cmap="gray", alpha=0.5)
    axes[1].imshow(vessel_mask_processed, cmap='Reds', alpha=0.7)
    axes[1].set_title("Vessel Mask (256x256)")
    axes[1].axis("off")
    
    # U-Net segmentation
    axes[2].imshow(original_processed, cmap="gray", alpha=0.5)
    im = axes[2].imshow(prediction, cmap=custom_cmap, alpha=0.7)
    axes[2].set_title("U-Net Segmentation")
    axes[2].axis("off")
    
    # Add colorbar legend
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Background', 'Alveoli', 'Blood Vessels'])
    
    plt.tight_layout()
    
    # Save and show
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'unet_segmentation_{original_filename}.png'))
    plt.show()
    
    print(f"Visualization saved to: {os.path.join(OUTPUT_DIR, f'unet_segmentation_{original_filename}.png')}")

if __name__ == "__main__":
    test_filename = "r01_0121.rec.16bit.tif"
    show_segmentation(test_filename)