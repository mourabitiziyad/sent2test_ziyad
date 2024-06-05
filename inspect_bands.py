import rasterio

def inspect_image_metadata(image_path):
    with rasterio.open(image_path) as src:
        print("Metadata:", src.meta)
        print("Band Count:", src.count)
        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"Band {i} - min: {band.min()}, max: {band.max()}")

# Example usage
inspect_image_metadata('Milan LCZ42.tif')