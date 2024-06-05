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

# Ensure the image has exactly 10 bands
def ensure_10_bands(image, target_channels=10):
    current_channels = image.shape[0]
    if current_channels < target_channels:
        padding_shape = (target_channels - current_channels, image.shape[1], image.shape[2])
        padding = np.zeros(padding_shape, dtype=image.dtype)
        image = np.concatenate((image, padding), axis=0)
    elif current_channels > target_channels:
        image = image[:target_channels, :, :]
    return image

# Extract patches
def extract_patches(image, patch_size=128, step=128):
    # Ensure the image has exactly 10 bands
    image = ensure_10_bands(image, 10)

    # Transpose the image to shape (height, width, channels)
    image = image.transpose(1, 2, 0)
    
    # Define the window shape for patches
    window_shape = (patch_size, patch_size, image.shape[2])

    # Extract patches using view_as_windows
    B = view_as_windows(image, window_shape, step)
    
    # Reshape patches to the desired format
    patches = B.reshape(-1, window_shape[0], window_shape[1], window_shape[2])
    
    print(f"Extracted patches shape: {patches.shape}")
    
    return patches, image.shape

# Save the prediction as a GeoTIFF file
def save_prediction_as_tiff(prediction, profile, output_path):
    # Normalize the predicted image to range [0, 255]
    prediction = (prediction * 255).astype(np.uint8)
    
    profile.update(
        dtype=rasterio.uint8,
        count=prediction.shape[2]
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(prediction.shape[2]):
            dst.write(prediction[:, :, i], i + 1)

# Define input and output directories
input_dir = 'test_images'
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# List all TIFF files in the input directory
image_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.tiff') or file.endswith('.tif')]

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
    predictions = model.predict(patches, batch_size=16, verbose=1)

    print(f"Predictions shape: {predictions.shape}")

    # Set the downsampling factor based on the model architecture
    downsample_factor = patches.shape[1] // predictions.shape[1]  # Determine the downsampling factor
    patch_size_out = patches.shape[1] // downsample_factor  # Output patch size

    num_patches_y = padded_shape[0] // patches.shape[1]
    num_patches_x = padded_shape[1] // patches.shape[2]
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

    # Ensure the prediction is 3D
    if len(predicted_image.shape) == 2:
        predicted_image = np.expand_dims(predicted_image, axis=2)

    # Save the prediction as a GeoTIFF file
    output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_prediction.tif')
    save_prediction_as_tiff(predicted_image, profile, output_path)

    # Visualize the prediction
    plt.figure(figsize=(10, 10))
    plt.imshow(predicted_image.squeeze(), cmap='gray')
    plt.colorbar(label='Prediction')
    plt.title(f'Predicted Segmentation for {os.path.basename(image_path)}')
    plt.show()
