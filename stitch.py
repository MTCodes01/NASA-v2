import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from reproject import reproject_interp
import glob
import os
from datetime import datetime

# --- Configuration ---
INTERMEDIATE_DIR = "intermediate_mosaics"
FINAL_OUTPUT = "tess_final_mosaic_from_intermediates.fits"

def stitch_intermediate_mosaics():
    """
    Stitches all intermediate mosaic files into a final mosaic.
    """
    print(f"\n{'='*70}")
    print(f"INTERMEDIATE MOSAIC STITCHER")
    print(f"{'='*70}")
    
    # Find all intermediate mosaic files
    intermediate_pattern = os.path.join(INTERMEDIATE_DIR, "intermediate_mosaic_*.fits")
    intermediate_files = sorted(glob.glob(intermediate_pattern))
    
    print(f"Found {len(intermediate_files)} intermediate mosaic files:")
    for file in intermediate_files:
        print(f"  - {os.path.basename(file)}")
    
    if len(intermediate_files) < 1:
        print("ERROR: No intermediate mosaic files found.")
        print(f"Looking for pattern: {intermediate_pattern}")
        return False
    
    start_time = datetime.now()
    
    # Load all intermediate mosaics
    input_data_wcs = []
    for filename in intermediate_files:
        try:
            with fits.open(filename) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                
                # Validate WCS
                if wcs.naxis != 2 or not wcs.has_celestial:
                    print(f"WARNING: Skipping {os.path.basename(filename)} - invalid WCS")
                    continue
                
                # Check for valid data
                if data is None or np.all(np.isnan(data)):
                    print(f"WARNING: Skipping {os.path.basename(filename)} - no valid data")
                    continue
                    
                input_data_wcs.append((data, wcs))
                print(f"✓ Loaded {os.path.basename(filename)} - Shape: {data.shape}")

        except Exception as e:
            print(f"ERROR loading {os.path.basename(filename)}: {e}")
            continue

    if len(input_data_wcs) < 1:
        print("ERROR: No valid intermediate mosaics to combine.")
        return False

    print(f"\n  Successfully loaded {len(input_data_wcs)} intermediate mosaics")
    
    # Calculate optimal WCS for final mosaic
    print(f"  Calculating optimal WCS for final mosaic...")
    try:
        reference_wcs, shape_out = find_optimal_celestial_wcs(
            input_data_wcs,
            frame='icrs', 
            projection='TAN'
        )
        print(f"  Final mosaic size: {shape_out}")
    except Exception as e:
        print(f"ERROR calculating optimal WCS: {e}")
        return False

    # Create final mosaic
    print(f"  Reprojecting and combining intermediate mosaics...")
    try:
        mosaic_array, footprint = reproject_and_coadd(
            input_data_wcs,
            reference_wcs,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function='mean'
        )
    except Exception as e:
        print(f"ERROR during reprojection: {e}")
        return False

    # Save final mosaic
    try:
        hdu = fits.PrimaryHDU(data=mosaic_array, header=reference_wcs.to_header())
        hdu.writeto(FINAL_OUTPUT, overwrite=True)
        
        elapsed = datetime.now() - start_time
        print(f"\n✓ FINAL MOSAIC SAVED: {FINAL_OUTPUT}")
        print(f"  Processing time: {elapsed}")
        print(f"  Final size: {mosaic_array.shape}")
        
    except Exception as e:
        print(f"ERROR saving final mosaic: {e}")
        return False
    
    return True

def list_intermediate_files():
    """
    Lists all available intermediate mosaic files.
    """
    intermediate_pattern = os.path.join(INTERMEDIATE_DIR, "intermediate_mosaic_*.fits")
    intermediate_files = sorted(glob.glob(intermediate_pattern))
    
    print(f"\nAvailable intermediate mosaic files:")
    print(f"Directory: {INTERMEDIATE_DIR}")
    print(f"Pattern: intermediate_mosaic_*.fits")
    print(f"Found: {len(intermediate_files)} files\n")
    
    if intermediate_files:
        for i, file in enumerate(intermediate_files, 1):
            try:
                with fits.open(file) as hdul:
                    shape = hdul[0].data.shape if hdul[0].data is not None else "No data"
                    print(f"  {i:2d}. {os.path.basename(file)} - Shape: {shape}")
            except Exception as e:
                print(f"  {i:2d}. {os.path.basename(file)} - ERROR: {e}")
    else:
        print("  No intermediate files found!")
        print(f"  Make sure the directory '{INTERMEDIATE_DIR}' exists")
        print(f"  and contains files matching 'intermediate_mosaic_*.fits'")

def main():
    """
    Main function - can be used standalone or imported.
    """
    print(f"\n{'='*70}")
    print(f"TESS INTERMEDIATE MOSAIC STITCHER")
    print(f"{'='*70}")
    
    # List available files first
    list_intermediate_files()
    
    # Ask for confirmation
    response = input(f"\nProceed with stitching? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Perform stitching
    success = stitch_intermediate_mosaics()
    
    if success:
        print(f"\n{'='*70}")
        print(f"SUCCESS: Final mosaic created!")
        print(f"Output file: {FINAL_OUTPUT}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"FAILED: Could not create final mosaic")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
