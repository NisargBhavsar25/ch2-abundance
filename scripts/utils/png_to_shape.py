import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd
import numpy as np
from rasterio.transform import from_bounds
import os
from pathlib import Path
import glob

def png_to_shapefile(input_png, output_shp):
    """
    Convert a PNG elemental map to a georeferenced shapefile.
    
    Parameters:
    input_png (str): Path to input PNG file
    output_shp (str): Path to output shapefile
    """
    # Read the PNG file
    with rasterio.open(input_png) as src:
        image = src.read(1)  # Read first band
        
        # Create the transform for proper georeferencing
        # For Moon coordinates: latitude (-90 to 90) and longitude (-180 to 180)
        transform = from_bounds(-180, -90, 180, 90, image.shape[1], image.shape[0])
        
        # Create a mask for valid data (non-zero pixels)
        mask = image > 0
        
        # Generate shapes from the raster
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(image, mask=mask, transform=transform))
        )
        
        # Convert the results to a list
        geoms = list(results)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geoms)
        
        # Set the coordinate reference system (CRS)
        # Using a simple spherical moon CRS
        gdf.crs = '+proj=longlat +R=1737400'  # Moon's radius in meters
        
        # Simplify geometries if needed while preserving topology
        gdf['geometry'] = gdf['geometry'].simplify(0.001, preserve_topology=True)
        
        # Save to shapefile
        gdf.to_file(output_shp)
        
        return gdf

def process_directory(input_dir, output_dir):
    """
    Process all PNG files in a directory and convert them to shapefiles.
    
    Parameters:
    input_dir (str): Directory containing PNG files
    output_dir (str): Directory to save output shapefiles
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files in the input directory
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    results = {}
    total_files = len(png_files)
    
    print(f"Found {total_files} PNG files to process")
    
    for i, png_file in enumerate(png_files, 1):
        # Get the filename without extension for naming the output
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        output_shp = os.path.join(output_dir, f"{base_name}.shp")
        
        print(f"Processing file {i}/{total_files}: {base_name}")
        
        try:
            results[base_name] = png_to_shapefile(png_file, output_shp)
            print(f"Successfully created: {output_shp}")
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
    
    print("\nProcessing complete!")
    print(f"Successfully processed: {len(results)}/{total_files} files")
    return results

# Example usage
if __name__ == "__main__":
    # Specify your input and output directories
    input_directory = "path/to/your/png/files"
    output_directory = r"path/to/your/shapefiles"
    
    # Process all PNG files in the directory
    results = process_directory(input_directory, output_directory)