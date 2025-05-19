# Encoder-decoder-1.0
This repository contains the code for the Master's Thesis *Title* written by Adam Svedberg and Sebastian Petersson on the Institution of Mathematical Statistics at the Faculty of Engineering at Lund University. 

The file test.py in the main folder handles pre-processesing and transforms volumes by size $2560 \times 2560 \times 2160$ into $1024 \times 1024 \times 1024$ as well as downsampling the volumes from 16 bit to 8 bit depth. It then saves these volumes to the folder /3d-stacks/.

In the file FMM_segmentation/main.py the active volume to segment can be chosen at the top of the main-function (line 102). When running main.py it then collects seed points from the folder /seed_points/ according to matching name (i.e. for r01_ it collects the file r01_seed_points.csv) and starts the FMM segmentation from these seed points. It expects the seed_points to have four columns in the order (index, x, y, z) where index is not used. It has some flexibility regarding columns.

By running ML_segmentation/train.py