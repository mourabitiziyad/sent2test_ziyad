import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

def normalize_image(image):
    # Clip to avoid extreme outliers
    image = np.clip(image, 0, 65535)
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def plot_image(image, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# Define the directory and get the list of images
input_dir = 'test_images'
image_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.tiff')]

# Load and plot each image
for image_path in image_paths:
    image, profile = load_image(image_path)
    
    # Select bands for visualization (typically RGB)
    if image.shape[0] >= 4:  # Check if there are at least 4 bands
        rgb_image = np.stack((image[3], image[2], image[1]), axis=-1)  # Sentinel-2 Band 4 (Red), Band 3 (Green), Band 2 (Blue)
    else:
        rgb_image = np.moveaxis(image, 0, 2)  # Fallback to using whatever bands are available

    # Normalize the image to [0, 1] range for visualization
    rgb_image = normalize_image(rgb_image)

    plot_image(rgb_image, title=f'Input Image: {os.path.basename(image_path)}')

    # Print the image profile
    print(f'Profile of {os.path.basename(image_path)}:')
    print(profile)
