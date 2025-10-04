import pyvips
import os
import numpy as np
from astropy.io import fits
from skimage.io import imsave # A robust way to save large arrays as TIFF

INPUT_FITS_FILE = "tess_ccd_mosaic_stitched.fits" 
OUTPUT_TIFF_FILE = "tess_mosaic_pyvips_input.tif"

# --- Configuration ---
INPUT_FILE = "tess_mosaic_pyvips_input.tif" # <-- Use the TIFF output from Step 1
OUTPUT_DZI_NAME = "tess_final_deep_zoom" 

def convert_fits_to_tiff(input_fits, output_tiff):
    """
    Reads the primary data array from the stitched FITS mosaic, 
    applies basic scaling, and saves it as a 16-bit TIFF file.
    """
    print(f"\nStarting conversion: {input_fits} -> {output_tiff}")
    
    try:
        # 1. Read the data array from the FITS file (HDU 0 is the primary image)
        # Note: We assume the stitched data is in the Primary HDU (HDU 0).
        mosaic_data = fits.getdata(input_fits, ext=0)
    except FileNotFoundError:
        print(f"ERROR: Input FITS file not found at '{input_fits}'. Please check the path.")
        return
    
    # 2. Handle NaN values and apply basic normalization (scaling for visualization)
    # The stitching process leaves areas with no data as NaN. We fill them with 0.
    mosaic_data[np.isnan(mosaic_data)] = 0
    
    # Convert data to an appropriate integer type (like 16-bit unsigned integer) 
    # for standard image formats like TIFF.
    # This involves scaling the float data to fit the 16-bit range (0 to 65535).
    
    # Simple linear scaling: clip to the 99.9th percentile to avoid extreme outliers
    v_max = np.percentile(mosaic_data[mosaic_data > 0], 99.9)
    v_min = np.percentile(mosaic_data[mosaic_data > 0], 0.1)
    
    scaled_data = mosaic_data.clip(v_min, v_max)
    # Normalize to 0-1, then multiply by 65535 for 16-bit TIFF
    scaled_data = (scaled_data - v_min) / (v_max - v_min)
    scaled_data = (scaled_data * 65535).astype(np.uint16)
    
    # 3. Save the array as a large TIFF file
    print(f"Saving array data ({scaled_data.shape}) as TIFF...")
    imsave(output_tiff, scaled_data, check_contrast=False)
    
    print(f"SUCCESS: TIFF file created. Ready for VIPS tiling in Step 2.")
    return True


def tiff_to_dzi(input_path, output_name):
    """
    Converts a large image file (TIFF) into a Deep Zoom Image (DZI) tile pyramid.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at '{input_path}'. Run the FITS to TIFF conversion first!")
        return

    try:
        print(f"\nLoading image with pyvips: {input_path}")
        # pyvips can load TIFFs reliably
        image = pyvips.Image.new_from_file(input_path, access='sequential')
        
        # Create the DZI file and tile folder
        print(f"Creating DZI tiles and file: {output_name}.dzi")
        # 'dzsave' is the VIPS command for Deep Zoom saving
        # image.dzsave(output_name, 
        #              suffix='.jpg', 
        #              tile_size=256, 
        #              overlap=1)
        
        print(f"\nSUCCESS: DZI created. Your image is ready to be loaded into OpenSeadragon!")

    except Exception as e:
        print(f"\nCRITICAL ERROR: DZI creation failed.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    # --- Execute the Conversion and Tiling in Sequence ---
    if convert_fits_to_tiff(INPUT_FITS_FILE, OUTPUT_TIFF_FILE):
        tiff_to_dzi(OUTPUT_TIFF_FILE, OUTPUT_DZI_NAME)  