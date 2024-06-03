# Satellite Image Processing and Visualization

This project provides scripts for processing and visualizing satellite images using Python. The scripts utilize libraries such as `rasterio`, `matplotlib`, `numpy`, and `keras` for image processing, plotting, and building a Convolutional Neural Network (CNN) model.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Script Descriptions](#script-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- Project built on Python 3.11+
- `rasterio`
- `matplotlib`
- `numpy`
- `keras`
- `scikit-image`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mourabitiziyad/sent2test_ziyad.git
    cd sent2test_ziyad
    ```

2. Create a virtual environment and activate it (optional):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Image Processing and Visualization

The script `main.py` is used to load, normalize, and visualize satellite images from a specified directory.

1. Update the `test_images` dir with the TIFF images.
2. Run `main.py`

## Input Visualisation

`input_image_vis.py` is used to quickly look at sentinel-2 input image. Running `input_image_vis.py` will automatically load each image in `test_images` in matplotlib. (Close the matplotlib window to open the next image)
   
