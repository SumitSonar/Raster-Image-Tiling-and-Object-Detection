import rasterio
import numpy as np
import os
from rasterio.windows import Window

# Define tile size
TILE_SIZE = 256  # Adjust as needed

# Create output directory
os.makedirs("tiles", exist_ok=True)

# Load the large TIFF image
image_path = "aircraft_image.tif"
with rasterio.open(image_path) as src:
    width, height = src.width, src.height  # Full image size
    profile = src.profile  # Save geospatial metadata
    
    # Loop to create tiles
    tile_id = 0
    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):
            window = Window(x, y, TILE_SIZE, TILE_SIZE)
            tile = src.read(window=window)

            # Convert multi-band to 3-channel RGB (if needed)
            if tile.shape[0] > 3:
                tile = tile[:3, :, :]  # Select first 3 bands
            
            # Normalize and save tile
            tile = (255 * (tile / tile.max())).astype(np.uint8)
            tile = np.transpose(tile, (1, 2, 0))  # Convert to (H, W, C)

            tile_filename = f"tiles/tile_{tile_id}.png"
            rasterio.open(
                tile_filename, 'w', driver='PNG', height=TILE_SIZE, width=TILE_SIZE, count=3, dtype='uint8'
            ).write(np.transpose(tile, (2, 0, 1)))

            tile_id += 1

print(f"✅ Image tiled into {tile_id} tiles")
