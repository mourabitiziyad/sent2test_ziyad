import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, AveragePooling2D, Concatenate, Dropout
import rasterio
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows

def sen2IS_net_bn_core(inputs, dim=16, inc_rate=2, bn=1):
    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    if bn == 1:
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)
    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(conv0)
    if bn == 1:
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)

    dim = dim * inc_rate
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(conv0)
    if bn == 1:
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    if bn == 1:
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D((2, 2))(conv1)
    pool2 = AveragePooling2D((2, 2))(conv1)
    merge1 = Concatenate()([pool1, pool2])

    return merge1

def sen2IS_net_bn_core_2(merge1, dim=128, inc_rate=2, numC=2, bn=1, attentionS=0, attentionC=0):
    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(merge1)
    if bn == 1:
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)
    if bn == 1:
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    drop0 = Dropout(0.1)(conv2)

    dim = dim * inc_rate
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(drop0)
    if bn == 1:
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)
    if bn == 1:
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    drop1 = Dropout(0.1)(conv3)

    o = Conv2D(numC, (1, 1), padding='same', activation='softmax')(drop1)
    return o

def sen2IS_net_bn(input_size=(128, 128, 10), numC=2, ifBN=0, attentionS=0, attentionC=0):
    inputs = Input(input_size)
    merge1 = sen2IS_net_bn_core(inputs, bn=ifBN)
    o = sen2IS_net_bn_core_2(merge1, dim=128, inc_rate=2, numC=numC, bn=ifBN, attentionS=attentionS, attentionC=attentionC)
    model = Model(inputs=inputs, outputs=o)
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

# Pad image channels to 10
def pad_image_channels(image, target_channels=10):
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
    # Transpose the image to shape (height, width, channels)
    image = image.transpose(1, 2, 0)
    
    # Define the window shape for patches
    window_shape = (patch_size, patch_size, image.shape[2])

    # Extract patches using view_as_windows
    B = view_as_windows(image, window_shape, step)
    
    # Reshape patches to the desired format
    patches = B.reshape(-1, window_shape[0], window_shape[1], window_shape[2])
    
    # Get the dimensions of the final map
    R = B.shape[0]
    C = B.shape[1]
    
    print(f"Extracted patches shape: {patches.shape}")
    
    return patches, R, C

# Save the prediction as a GeoTIFF file
def save_prediction_as_tiff(prediction, profile, output_path):
    # Check if the prediction is single-channel or multi-channel
    if len(prediction.shape) == 2:
        profile.update(dtype=rasterio.uint8, count=1)
        prediction = np.expand_dims(prediction, axis=2)
    else:
        profile.update(dtype=rasterio.uint8, count=prediction.shape[2])

    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(prediction.shape[2]):
            dst.write(prediction[:, :, i], i + 1)

def pro_from_x(mapR, mapC, y, padding):
    mapPatch_shape = y.shape[1]
    B_ = np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2], y.shape[-1]))
    C = np.zeros((B_.shape[0] * B_.shape[2], B_.shape[1] * B_.shape[3], B_.shape[4]), dtype=float)
    for dim in np.arange(B_.shape[4]):
        B_1 = B_[:, :, :, :, dim]
        C[:, :, dim] = B_1.transpose(0, 2, 1, 3).reshape(-1, B_1.shape[1] * B_1.shape[3])
    return C, mapPatch_shape

# Define input and output directories
input_dir = 'test_images'
output_dir = 'new_output_images'
os.makedirs(output_dir, exist_ok=True)

# List all TIFF files in the input directory
image_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.tiff')]

# Define the patch size and padding list
patch_shape = (128, 128, 10)
step = patch_shape[0]
params = {'dim_x': patch_shape[0],
          'dim_y': patch_shape[1],
          'dim_z': patch_shape[2],
          'step': step,
          'Bands': [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
          'scale': 10000.0,
          'nanValu': 999}

dim_x = params['dim_x']
if dim_x == 32:
    paddList = [0, 8, 16, 24]
elif dim_x == 64:
    paddList = [0, 16, 32, 48]
else:
    paddList = [0, 32, 64, 96]

# Process each image
for image_path in image_paths:
    print(f"Processing {image_path}")

    # Load the image
    images, profile = load_image(image_path)

    # Normalize the image
    normalized_images = normalize_image(images)
    
    # Pad the image channels to 10
    padded_images = pad_image_channels(normalized_images)

    Pro = None

    # Process each padding
    for padding in paddList:
        if padding == 0:
            img1 = padded_images
        else:
            img1 = np.pad(padded_images, ((0, 0), (padding, 0), (padding, 0)), 'reflect')

        print(f"Padded shape: {img1.shape}")

        # Extract patches
        patches, mapR, mapC = extract_patches(img1, patch_size=dim_x, step=dim_x)

        # Predict using the pre-trained model
        predictions = model.predict(patches, batch_size=16, verbose=1)

        print(f"Predictions shape: {predictions.shape}")

        # Process predictions to match the original method
        C, mapPatch_shape = pro_from_x(mapR, mapC, predictions, padding)
        OS = int(dim_x / mapPatch_shape)

        if padding == 0:
            r, c = C.shape[:2]
            Pro = C[:r - mapPatch_shape, :c - mapPatch_shape, :]
        else:
            Pro += C[int(padding / OS):r - mapPatch_shape + int(padding / OS), int(padding / OS):c - mapPatch_shape + int(padding / OS), :]

    # Save the prediction as a GeoTIFF file
    output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_predicted.tif')
    save_prediction_as_tiff(Pro, profile, output_path)
