# Encoder-Decoder and Fast Marching for segmentation of blood vessels in lung tissue
This repository contains the code for the Master's Thesis *Semi-supervised segmentation of blood vessels in synchrotron μm-CT scans* written by **Adam Svedberg** and **Sebastian Petersson** at the Division of Mathematical Statistics at the Faculty of Engineering at Lund University. The code is used to create masks for blood vessels via seed points and *Constrained-FMM* and then a final segmentation using an *Attention ResU-net*. Pre-trained weights for the ML model, images and other information can be found [here](https://drive.google.com/drive/folders/1R0lQdSKNx96qXUxH2xhvGxUfSIvxlVTM?usp=sharing).

## General information about the code and it's operations.

The file pre-process.py in the main folder handles pre-processesing and transforms volumes by size $2560 \times 2560 \times 2160$ into $1024 \times 1024 \times 1024$ as well as downsampling the volumes from 16 bit to 8 bit depth. It then saves these volumes to the folder "3d-stacks/". It collects the original volumes from "images/{file_name}/rec_16bit_Paganin_0/". Towards the end of the code (row ~80) it can be instructed to go through all subfolders with images in the folder "images/" or certain file_names in the folder.

In the file "FMM_segmentation/main.py" the active volume to segment can be chosen at the top of the main-function (line 102). When running main.py it then collects seed points from the folder "seed_points/" according to matching name (i.e. for r01_ it collects the file r01_seed_points.csv) and starts the FMM segmentation from these seed points. It expects the seed_points to have four columns in the order (index, x, y, z) where index is not used. It has some flexibility regarding columns. 

By running "ML_segmentation/train.py" the ML model is trained on the data it is instructed to use via "ML_segmentation/configuration.py" (~line 14). It collects the (pre-processed) images from the folder "3d-stacks/{file_name}/", and the masks from the folder "results/". It then saves three models in the folder "checkpoints/", the one from the last epoch, the one with the best recall metric and the one with the best validation loss (it compares the saved models after each epoch and updates which one to save). 

In order to visualise and evaluate a trained model the file "ML_segmentation/orthagonal_evaluation.py" should be run. This program evaluates volumes accross the x, y, and z-axes and forms a combined evaluation. It can be prompted to perform for one file by parsing --file {file_name} or for all files in "3d-stacks/" by parsing --all. By parsing --model {path_to_model} you can choose which model you want to use, otherwise the model with the best validation loss is used as default, saved in "checkpoints/". The programs output is saved under "evaluation_results/" with a timestamp in order to distinguish different evaluations from one another. It saves a binary segmentation as a .nii.gz-file, a probabilty map as a .tif-file and a plot of the segmentation overlayed across the original image for the three middle slices as a .png for each file.

In the file visualize_results.py a visualisation of the three middle slices for the FMM-segmentations is created, similarily to the comparison created by "orthagonal_evaluation.py". These files are saved in "middle_slice_visualizations_{date_and_time}".

In "evaluation_metrics/compare_segmentations.py" evaluation metrics and comparisons of the FMM- and ML-segmentations are created.

The code is written in Python and mainly uses Tensorflow for ML and SimpleITK for FMM.

## System Requirements and Hardware Specifications

This project was developed and tested on the following hardware configuration:
- **GPUs**: 2× Nvidia Tesla V100-PCIE-16GB
- **CPU**: Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz
- **RAM**: 125 GB 
- **Storage**: At least 600 GB recommended for dataset storage
- **CUDA Version**: 12.2 
- **OS**: Linux Mint 21.2

### Software Dependencies
- Python        3.10.12
- TensorFlow    2.19
- SimpleITK     2.4.1
- NumPy         2.1.3
- scikit-image  0.25.2
- matplotlib    3.10.1
- pandas        2.2.3
- nibabel       5.3.2

### Computational Requirements
- Processing the full dataset requires approximately 15 GB of GPU memory
- Training time: Approximately 8 minutes of startup and then 13 minutes per epoch on the specified hardware
- Evaluation of a single volume takes approximately 6 minutes

To install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Storage Requirements
- Original dataset: ~432 GB
- Pre-processed volumes: ~16 GB
- Segmentation results: ~64 GB

For best performance, we recommend running this code on a machine with at least one CUDA-capable GPU with 16 GB of VRAM.

## Project Structure

```
Encoder-Decoder/
├── 3d-stacks/             # Pre-processed 3D volumes
├── FMM_segmentation/      # Fast Marching Method implementation
├── ML_segmentation/       # Machine Learning model implementation
├── checkpoints/           # Saved ML models
├── evaluation_metrics/    # Code for comparing segmentation results
│   └── comparison_results/# Visual and statistical comparisons
├── evaluation_results/    # ML model evaluation outputs
├── images/                # Original input CT volumes
├── results/               # FMM segmentation results
├── seed_points/           # Input seed points for FMM
├── pre-process.py         # Data preparation script
├── visualize_results.py   # Visualization utilities
└── requirements.txt       # Package dependencies
```

## Installation and Setup

1. Clone this repository:
```bash
git clone https://github.com/sebsgitc/Encoder-Decoder.git
cd Encoder-Decoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your dataset to the `images/` directory following the structure described above.

## Usage Example

### Data Preprocessing
```bash
python pre-process.py
```

### FMM Segmentation
```bash
python FMM_segmentation/main.py
```

### ML Model Training
```bash
python ML_segmentation/train.py
```

### ML Model Evaluation
```bash
python ML_segmentation/orthagonal_evaluation.py --all
# Or for a specific file
python ML_segmentation/orthagonal_evaluation.py --file r01_
```

### Comparing Results
```bash
python evaluation_metrics/compare_segmentations.py
```

## Citing This Work

If you use this code in your research, please cite our thesis:

```
@mastersthesis{svedberg2025segmentation,
  author       = {Svedberg, Adam and Petersson, Sebastian},
  title        = {Title},
  school       = {Lund University},
  year         = {2025},
  month        = {June}
}
```

## License

This project is licensed under the MIT license.

## Contact

For questions about the code, please contact:
- Adam Svedberg - [ad4631sv-s@student.lu.se]
- Sebastian Petersson - [se3475pe-s@student.lu.se]