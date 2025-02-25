import os

# Define project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))

# Paths to images & masks
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
MASK_DIR = os.path.join(PROJECT_DIR, "masks")

# Image size
IMG_HEIGHT = 1024
IMG_WIDTH = 1024

#Minibatch size
MB_SIZE = 2
