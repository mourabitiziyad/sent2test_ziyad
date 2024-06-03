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

print('Model loaded successfully!')

# Function to load specific bands from Sentinel-2 images
def load_image(image_path_pattern, bands):
    images = []
    profile = None
    for band in bands:
        with rasterio.open(image_path_pattern.replace('BAND', band)) as src:
            images.append(src.read(1))
            if profile is None:
                profile = src.profile
    images = np.stack(images, axis=-1)
    return images, profile

# Specify the bands to be used
bands = ['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8a', 'B11', 'B12']

# Path pattern to your Sentinel-2 images, where 'BAND' will be replaced with the actual band identifier
image_path_pattern = 'bands/2024-04-30-00:00_2024-04-30-23:59_Sentinel-2_L2A_BAND_(Raw).tiff'

# Load the images
images, profile = load_image(image_path_pattern, bands)

# Normalize the image
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

normalized_images = normalize_image(images)

# Extract patches
patch_size = 128  # Adjust based on model input size
step = patch_size
patches = view_as_windows(normalized_images, (patch_size, patch_size, 10), step)
patches = patches.reshape(-1, patch_size, patch_size, 10)

# Predict using the pre-trained model
predictions = model.predict(patches)

# Reshape predictions to the original image shape
predicted_image = predictions.reshape((normalized_images.shape[0] // patch_size,
                                       normalized_images.shape[1] // patch_size, patch_size, patch_size))

predicted_image = np.block([[predicted_image[i, j] for j in range(predicted_image.shape[1])] for i in range(predicted_image.shape[0])])

# Save the prediction as a GeoTIFF file
def save_prediction_as_tiff(prediction, profile, output_path):
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(rasterio.float32), 1)

output_path = 'predicted_segmentation.tif'  # Update with desired output path
save_prediction_as_tiff(predicted_image, profile, output_path)

# Visualize the prediction
plt.figure(figsize=(10, 10))
plt.imshow(predicted_image, cmap='gray')
plt.colorbar(label='Prediction')
plt.title('Predicted Segmentation')
plt.show()
