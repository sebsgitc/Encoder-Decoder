import os

# Define project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))

# Paths to images & masks
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
MASK_DIR = os.path.join(PROJECT_DIR, "masks")

# Image size
IMG_HEIGHT = 512
IMG_WIDTH = 512
