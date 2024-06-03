import rasterio
import numpy as np
import matplotlib.pyplot as plt

# File paths for NIR and SWIR images
nir_path = 'NIR_image.B8A.tif'
swir_path = 'SWIR_image.B11.tif'
output_path = 'ndbi_result_12_pro.tif'

# Open the NIR and SWIR images
# with rasterio.open(nir_path) as nir_src:
#     nir = nir_src.read(1)
#     meta = nir_src.meta

# with rasterio.open(swir_path) as swir_src:
#     swir = swir_src.read(1)

# # Calculate NDBI
# ndbi = (swir - nir) / (swir + nir)

# # Update the metadata for the output file
# meta.update(dtype=rasterio.float32, count=1)

# # Save the NDBI result to a new GeoTIFF file
# ndbi_output_path = 'ndbi_result.tif'
# with rasterio.open(ndbi_output_path, 'w', **meta) as dst:
#     dst.write(ndbi.astype(rasterio.float32), 1)



# Plot the NDBI
with rasterio.open(output_path) as src:
    ndbi = src.read(1)


plt.figure(figsize=(10, 10))
plt.imshow(ndbi, cmap='gray')
plt.colorbar(label='NDBI')
plt.title('Normalized Difference Built-up Index (NDBI)')
plt.show()
