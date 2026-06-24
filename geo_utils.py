from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape


def read_tiff_as_rgb(tiff_path):
    with rasterio.open(tiff_path) as src:
        img = src.read([1, 2, 3]).astype(np.float32)
        transform = src.transform
        crs = src.crs

    img = np.transpose(img, (1, 2, 0))
    return img, transform, crs


def stretch_for_display(img):
    """Only for visualization in Streamlit. Does not affect YOLO input."""
    img = img.astype(np.float32)
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    img = (img - p2) / (p98 - p2 + 1e-6)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def norm_band_like_kaggle(band):
    """Same band-wise min-max normalization used in Kaggle patch generation."""
    band = band.astype(np.float32)
    return ((band - band.min()) / (band.max() - band.min() + 1e-5) * 255).astype(np.uint8)


def create_kaggle_style_tif_patches(tiff_path, output_dir, patch_size=128, overlap=30, prefix="patch"):
    """
    Create saved georeferenced 128x128 TIFF patches using the same strategy as Kaggle:
    - read bands 1/2/3
    - normalize each band separately by min-max
    - save as uint8 GeoTIFF preserving CRS and window transform
    Returns: output_dir, patch_records
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = patch_size - overlap
    patch_records = []
    count = 0

    with rasterio.open(tiff_path) as src:
        for y in range(0, src.height - patch_size + 1, stride):
            for x in range(0, src.width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                transform = src.window_transform(window)

                red = src.read(1, window=window)
                green = src.read(2, window=window)
                blue = src.read(3, window=window)

                rgb = np.stack([
                    norm_band_like_kaggle(red),
                    norm_band_like_kaggle(green),
                    norm_band_like_kaggle(blue),
                ])

                profile = src.profile.copy()
                profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "count": 3,
                    "dtype": rasterio.uint8,
                    "transform": transform,
                    "driver": "GTiff",
                })

                out_path = output_dir / f"{prefix}_patch_{count}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(rgb)

                patch_records.append({
                    "patch_no": count + 1,
                    "patch_path": str(out_path),
                    "x": x,
                    "y": y,
                    "width": patch_size,
                    "height": patch_size,
                })
                count += 1

    return output_dir, patch_records


def mask_to_polygons(mask, transform, crs):
    polygons = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask > 0, transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    if len(polygons) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    return gpd.GeoDataFrame(geometry=polygons, crs=crs)
