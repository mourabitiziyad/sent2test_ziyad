import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, AveragePooling2D, Concatenate, Dropout
import rasterio
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows

def sen2IS_net_bn(input_size=(128, 128, 10), num_classes=2):
    inputs = Input(input_size)
    
    # First block
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    
    # Second block
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    
    pool1 = MaxPooling2D((2, 2))(x)
    pool2 = AveragePooling2D((2, 2))(x)
    x = Concatenate()([pool1, pool2])
    
    # Third block
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    
    # Fourth block
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

# Instantiate the model
model = sen2IS_net_bn()

print(model.summary())

# Load the weights
model_path = '_12_8weights.best.hdf5'
model.load_weights(model_path)

print("Model loaded successfully.")

# Function to load specific bands from Sentinel-2 images
def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

# Normalize the image
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Extract patches
def extract_patches(image, patch_size=128):
    step = patch_size

    # Calculate padding
    pad_height = (patch_size - (image.shape[1] % patch_size)) % patch_size
    pad_width = (patch_size - (image.shape[2] % patch_size)) % patch_size

    print(f"Original shape: {image.shape}")
    print(f"Padding height: {pad_height}, Padding width: {pad_width}")

    # Pad the image
    padded_image = np.pad(image, 
                          ((0, 0), (0, pad_height), (0, pad_width)), 
                          mode='reflect')

    print(f"Padded shape: {padded_image.shape}")

    # Extract patches
    patches = view_as_windows(padded_image.transpose(1, 2, 0), (patch_size, patch_size, 10), step)
    patches = patches.reshape(-1, patch_size, patch_size, 10)

    print(f"Patches shape: {patches.shape}")

    return patches, padded_image.shape

# Save the prediction as a GeoTIFF file
def save_prediction_as_tiff(prediction, profile, output_path):
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction, 1)

# Define input and output directories
input_dir = 'test_images'
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# List all TIFF files in the input directory
image_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]

# Process each image
for image_path in image_paths:
    print(f"Processing {image_path}")
    
    # Load the image
    images, profile = load_image(image_path)

    # Normalize the image
    normalized_images = normalize_image(images)

    # Extract patches
    patches, padded_shape = extract_patches(normalized_images)

    # Predict using the pre-trained model
    predictions = model.predict(patches)

    print(f"Predictions shape: {predictions.shape}")

    # Set the downsampling factor based on the model architecture
    downsample_factor = patches.shape[1] // predictions.shape[1]  # Determine the downsampling factor
    patch_size_out = patches.shape[1] // downsample_factor  # Output patch size

    num_patches_y = padded_shape[1] // patches.shape[1]
    num_patches_x = padded_shape[2] // patches.shape[2]
    num_channels = predictions.shape[-1]

    print(f"Number of patches (y, x): {num_patches_y}, {num_patches_x}")
    print(f"Number of channels: {num_channels}")
    print(f"Output patch size: {patch_size_out}")

    # Verify the total number of elements
    total_elements = num_patches_y * num_patches_x * patch_size_out * patch_size_out * num_channels
    print(f"Total elements expected: {total_elements}")
    print(f"Total elements in predictions: {predictions.size}")

    assert total_elements == predictions.size, "Mismatch in the total number of elements for reshaping."

    # Reshape predictions to the padded image shape
    predicted_padded_image = predictions.reshape((num_patches_y, num_patches_x, patch_size_out, patch_size_out, num_channels))

    # Reconstruct the predicted image
    predicted_image = np.block([[predicted_padded_image[i, j, :, :, 0] for j in range(predicted_padded_image.shape[1])] for i in range(predicted_padded_image.shape[0])])

    # Remove padding
    predicted_image = predicted_image[:normalized_images.shape[1], :normalized_images.shape[2]]

    # Apply a threshold to the predicted image to get binary segmentation
    thresholds = [0.001, 0.01, 0.1]
    for threshold in thresholds:

        binary_predicted_image = (predicted_image > threshold).astype(np.uint8)

        # Save the prediction as a GeoTIFF file
        output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_binary_predicted_segmentation_threshold_{threshold}.tif')
        save_prediction_as_tiff(binary_predicted_image, profile, output_path)

        # Visualize the prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_predicted_image, cmap='gray')
        plt.colorbar(label='Prediction')
        plt.title(f'Binary Predicted Segmentation (Threshold: {threshold})')
        plt.show()
