import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
import numpy as np
import glob
import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def combine_rasters(reflectance_tif, xrf_tifs_dir, output_tif='combined_map.tif', plot=False, cmap='viridis'):
    
    lunar_crs = CRS.from_proj4('+proj=longlat +a=1737400 +b=1737400 +no_defs')

    try:
        with rasterio.open(reflectance_tif) as reflectance_dataset:
            reflectance_data = reflectance_dataset.read(1)  
            reflectance_transform = reflectance_dataset.transform
            reflectance_crs = reflectance_dataset.crs
            reflectance_bounds = reflectance_dataset.bounds
            reflectance_meta = reflectance_dataset.meta.copy()
        logger.info("Loaded lunar reflectance map.")
    except Exception as e:
        logger.error(f"Error loading reflectance TIFF file: {e}")
        raise

    xrf_tif_files = glob.glob(os.path.join(xrf_tifs_dir, '*.tif'))
    if not xrf_tif_files:
        logger.error(f"No TIFF files found in directory: {xrf_tifs_dir}")
        raise FileNotFoundError(f"No TIFF files found in directory: {xrf_tifs_dir}")
    logger.info(f"Found {len(xrf_tif_files)} XRF TIFF files.")

    xrf_datasets = []
    for tif in xrf_tif_files:
        try:
            ds = rasterio.open(tif)
            ds._crs = lunar_crs  
            xrf_datasets.append(ds)
        except Exception as e:
            logger.error(f"Error loading XRF TIFF file {tif}: {e}")
            continue
    logger.info("Assigned lunar CRS to XRF datasets.")

    try:
        xrf_merged_data, xrf_merged_transform = merge(xrf_datasets)
        logger.info("Merged XRF datasets.")
    except Exception as e:
        logger.error(f"Error merging XRF datasets: {e}")
        raise
    finally:
        for ds in xrf_datasets:
            ds.close()

    xrf_crs = lunar_crs
    logger.info("Assigned lunar CRS to xrf_crs.")

    if reflectance_crs is None:
        reflectance_crs = lunar_crs
        reflectance_meta.update({'crs': lunar_crs})
        logger.info("Assigned lunar CRS to reflectance_crs.")

    if xrf_crs is None:
        xrf_crs = lunar_crs
        logger.info("Assigned lunar CRS to xrf_crs.")

    if xrf_crs != reflectance_crs:
        dst_transform, width, height = calculate_default_transform(
            xrf_crs, reflectance_crs, xrf_merged_data.shape[2], xrf_merged_data.shape[1],
            *xrf_datasets[0].bounds)
        xrf_reprojected_data = np.empty((1, height, width), dtype=rasterio.float32)

        reproject(
            source=xrf_merged_data,
            destination=xrf_reprojected_data,
            src_transform=xrf_merged_transform,
            src_crs=xrf_crs,
            dst_transform=dst_transform,
            dst_crs=reflectance_crs,
            resampling=Resampling.nearest)

        xrf_data = xrf_reprojected_data
        xrf_transform = dst_transform
        logger.info("Reprojected XRF data to match lunar reflectance map CRS.")
    else:
        xrf_data = xrf_merged_data
        xrf_transform = xrf_merged_transform
        logger.info("CRS match, skipping reprojection.")

    xrf_resampled = np.empty(
        (xrf_data.shape[0], reflectance_data.shape[0], reflectance_data.shape[1]),
        dtype=rasterio.float32
    )


    reproject(
        source=xrf_data,
        destination=xrf_resampled,
        src_transform=xrf_transform,
        src_crs=reflectance_crs,
        dst_transform=reflectance_transform,
        dst_crs=reflectance_crs,
        resampling=Resampling.nearest
    )
    logger.info("Resampled XRF data to match lunar reflectance map grid.")

    xrf_resampled_band = xrf_resampled[0]

    combined_data = np.where(xrf_resampled_band != 0, xrf_resampled_band, reflectance_data)

    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        extent = rasterio.plot.plotting_extent(reflectance_dataset)
        cax = ax.imshow(combined_data, cmap=cmap, extent=extent)
        ax.set_title('XRF Coverage Mapped onto Lunar Reflectance Map')
        fig.colorbar(cax, ax=ax, label='Data Values')
        plt.show()

    combined_meta = reflectance_meta.copy()
    combined_meta.update({
        "driver": "GTiff",
        "height": combined_data.shape[0],
        "width": combined_data.shape[1],
        "transform": reflectance_transform
    })

    try:
        with rasterio.open(output_tif, 'w', **combined_meta) as dest:
            dest.write(combined_data, 1)
        logger.info(f"Saved the combined map as '{output_tif}'.")
    except Exception as e:
        logger.error(f"Error saving combined TIFF file: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Combine lunar reflectance map with XRF TIFF files.')
    parser.add_argument('reflectance_tif', type=str, help='Path to the lunar reflectance TIFF file.')
    parser.add_argument('xrf_tifs_dir', type=str, help='Directory containing XRF TIFF files.')
    parser.add_argument('--output_tif', type=str, default='combined_map.tif',
                        help='Path to the output combined TIFF file.')
    parser.add_argument('--plot', action='store_true', help='Plot the combined map.')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for plotting.')

    args = parser.parse_args()

    combine_rasters(args.reflectance_tif, args.xrf_tifs_dir, args.output_tif, args.plot, args.cmap)
