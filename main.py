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
INPUT_DIRECTORY = "tess_sector_97_ffis" 
FITS_FILE_PATTERN = os.path.join(INPUT_DIRECTORY, '*ffic.fits')
INTERMEDIATE_DIR = "intermediate_mosaics"
FINAL_OUTPUT = "tess_ccd_mosaic_stitched_final.fits"

# Batch size: adjust based on your RAM (100 files should be safe for 32GB)
BATCH_SIZE = 100

# --- Helper Function to Clean WCS ---
def get_2d_celestial_data(filename):
    """
    Reads a TESS FFI FITS file, extracts the 2D image data, 
    and returns a WCS object explicitly configured for 2D celestial coordinates.
    """
    with fits.open(filename, mode='readonly') as hdulist:
        valid_hdu = None
        for i, hdu in enumerate(hdulist):
            if hdu.data is not None and 'CTYPE1' in hdu.header:
                valid_hdu = i
                break
        
        if valid_hdu is None:
            raise ValueError(f"No HDU with valid WCS found in {filename}")
        
        hdu_index = valid_hdu
        data = hdulist[hdu_index].data
        header = hdulist[hdu_index].header
        
        # Handle multi-dimensional data first
        if data.ndim == 3:
            mid_cadence = data.shape[0] // 2
            data_2d = data[mid_cadence, :, :]
        elif data.ndim == 2:
            data_2d = data
        else:
            raise ValueError(f"Unexpected data dimensions: {data.ndim}")
        
        # Create WCS and check what we have
        wcs_full = WCS(header)
        wcs_2d = None
        
        # Approach 1: Use .celestial if it exists
        if wcs_full.has_celestial:
            try:
                wcs_2d = wcs_full.celestial
            except Exception:
                pass
        
        # Approach 2: Try sub-selecting specific axes
        if wcs_2d is None or not wcs_2d.has_celestial:
            ra_axis = None
            dec_axis = None
            
            for i in range(wcs_full.naxis):
                ctype = wcs_full.wcs.ctype[i]
                if 'RA' in ctype or 'LON' in ctype:
                    ra_axis = i
                elif 'DEC' in ctype or 'LAT' in ctype:
                    dec_axis = i
            
            if ra_axis is not None and dec_axis is not None:
                try:
                    wcs_2d = wcs_full.sub([ra_axis + 1, dec_axis + 1])
                except Exception:
                    pass
        
        # Approach 3: Build from scratch
        if wcs_2d is None or not wcs_2d.has_celestial or wcs_2d.naxis != 2:
            wcs_2d = WCS(naxis=2)
            
            celestial_axis_1 = None
            celestial_axis_2 = None
            
            for i in range(1, 10):
                ctype_key = f'CTYPE{i}'
                if ctype_key in header:
                    ctype_val = header[ctype_key]
                    if 'RA' in ctype_val or 'LON' in ctype_val:
                        celestial_axis_1 = i
                    elif 'DEC' in ctype_val or 'LAT' in ctype_val:
                        celestial_axis_2 = i
            
            if celestial_axis_1 is None or celestial_axis_2 is None:
                raise ValueError(f"Cannot find RA/Dec axes in header for {filename}")
            
            wcs_2d.wcs.ctype[0] = header.get(f'CTYPE{celestial_axis_1}', 'RA---TAN')
            wcs_2d.wcs.ctype[1] = header.get(f'CTYPE{celestial_axis_2}', 'DEC--TAN')
            wcs_2d.wcs.crval[0] = header.get(f'CRVAL{celestial_axis_1}', 0.0)
            wcs_2d.wcs.crval[1] = header.get(f'CRVAL{celestial_axis_2}', 0.0)
            wcs_2d.wcs.crpix[0] = header.get(f'CRPIX{celestial_axis_1}', 1.0)
            wcs_2d.wcs.crpix[1] = header.get(f'CRPIX{celestial_axis_2}', 1.0)
            
            cd_keys = [
                (f'CD{celestial_axis_1}_{celestial_axis_1}', f'CD{celestial_axis_1}_{celestial_axis_2}'),
                (f'CD{celestial_axis_2}_{celestial_axis_1}', f'CD{celestial_axis_2}_{celestial_axis_2}')
            ]
            
            if all(key in header for row in cd_keys for key in row):
                wcs_2d.wcs.cd = [
                    [header[cd_keys[0][0]], header[cd_keys[0][1]]],
                    [header[cd_keys[1][0]], header[cd_keys[1][1]]]
                ]
            elif f'CDELT{celestial_axis_1}' in header and f'CDELT{celestial_axis_2}' in header:
                wcs_2d.wcs.cdelt[0] = header[f'CDELT{celestial_axis_1}']
                wcs_2d.wcs.cdelt[1] = header[f'CDELT{celestial_axis_2}']
            
            if 'RADESYS' in header:
                wcs_2d.wcs.radesys = header['RADESYS']
            if 'EQUINOX' in header:
                wcs_2d.wcs.equinox = header['EQUINOX']
        
        if wcs_2d.naxis != 2 or not wcs_2d.has_celestial:
            raise ValueError(f"Failed to create valid 2D celestial WCS for {filename}")

    return data_2d, wcs_2d

# --- Batch Mosaicking Function ---
def stitch_batch(file_list, output_filename, batch_num, total_batches):
    """
    Stitches a batch of FITS files into a single intermediate mosaic.
    """
    print(f"\n{'='*70}")
    print(f"BATCH {batch_num}/{total_batches}: Processing {len(file_list)} files")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    
    input_data_wcs = []
    for filename in file_list:
        try:
            data, wcs = get_2d_celestial_data(filename)
            
            if wcs.naxis != 2 or not wcs.has_celestial:
                print(f"Skipping {os.path.basename(filename)} - invalid WCS")
                continue
                
            input_data_wcs.append((data, wcs))
            print(f"Loaded {os.path.basename(filename)}")
            
        except Exception as e:
            print(f"ERROR processing {os.path.basename(filename)}: {e}")

    if len(input_data_wcs) < 2:
        print(f"ERROR: Not enough valid files in batch {batch_num}. Skipping.")
        return False

    print(f"\n  Successfully loaded {len(input_data_wcs)}/{len(file_list)} files")
    print(f"  Calculating optimal WCS for batch...")
    
    reference_wcs, shape_out = find_optimal_celestial_wcs(
        input_data_wcs,
        frame='icrs', 
        projection='TAN'
    )
    print(f"  Batch mosaic size: {shape_out}")

    print(f"  Reprojecting and co-adding...")
    mosaic_array, footprint = reproject_and_coadd(
        input_data_wcs,
        reference_wcs,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function='mean'    
    )

    hdu = fits.PrimaryHDU(data=mosaic_array, header=reference_wcs.to_header())
    hdu.writeto(output_filename, overwrite=True)
    
    elapsed = datetime.now() - start_time
    print(f"Batch {batch_num} saved: {output_filename}")
    print(f"Time elapsed: {elapsed}")

    return True

# --- Final Mosaicking Function ---
def stitch_final_mosaic(intermediate_files, output_filename):
    """
    Stitches all intermediate mosaics into the final mosaic.
    """
    print(f"\n{'='*70}")
    print(f"FINAL STITCHING: Combining {len(intermediate_files)} intermediate mosaics")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    input_data_wcs = []
    for filename in intermediate_files:
        try:
            with fits.open(filename) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                
                if wcs.naxis != 2 or not wcs.has_celestial:
                    print(f"Skipping {os.path.basename(filename)} - invalid WCS")
                    continue
                    
                input_data_wcs.append((data, wcs))
                print(f"Loaded {os.path.basename(filename)}")

        except Exception as e:
            print(f"ERROR loading {os.path.basename(filename)}: {e}")

    if len(input_data_wcs) < 1:
        print("ERROR: No valid intermediate mosaics to combine.")
        return False

    print(f"\n  Calculating final optimal WCS...")
    reference_wcs, shape_out = find_optimal_celestial_wcs(
        input_data_wcs,
        frame='icrs', 
        projection='TAN'
    )
    print(f"  Final mosaic size: {shape_out}")

    print(f"  Creating final mosaic (this may take a while)...")
    mosaic_array, footprint = reproject_and_coadd(
        input_data_wcs,
        reference_wcs,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function='mean'    
    )

    hdu = fits.PrimaryHDU(data=mosaic_array, header=reference_wcs.to_header())
    hdu.writeto(output_filename, overwrite=True)
    
    elapsed = datetime.now() - start_time
    print(f"\nFINAL MOSAIC SAVED: {output_filename}")
    print(f"Time elapsed: {elapsed}")

    # Create preview
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection=reference_wcs)
    
    v_data = mosaic_array[~np.isnan(mosaic_array)]
    if len(v_data) > 0:
        v_min = np.percentile(v_data, 1)
        v_max = np.percentile(v_data, 99)
        
        ax.imshow(mosaic_array, 
                  vmin=v_min,
                  vmax=v_max,
                  origin='lower', 
                  cmap='gray')
    
    ax.set_xlabel('Right Ascension (RA)')
    ax.set_ylabel('Declination (Dec)')
    plt.title('TESS FFI Final Mosaic')
    plt.tight_layout()
    plt.savefig('tess_mosaic_final_preview.png', dpi=150, bbox_inches='tight')
    print(f"  Preview saved: tess_mosaic_final_preview.png")
    plt.close()
    
    return True

# --- Main Execution ---
def main():
    overall_start = datetime.now()
    
    # Create intermediate directory
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    
    # Get all FITS files
    all_files = sorted(glob.glob(FITS_FILE_PATTERN))
    print(f"\n{'='*70}")
    print(f"TESS FFI BATCH MOSAIC STITCHER")
    print(f"{'='*70}")
    print(f"Found {len(all_files)} FITS files")
    print(f"Batch size: {BATCH_SIZE} files per batch")
    
    if len(all_files) < 2:
        print("ERROR: Need at least 2 FITS files.")
        return
    
    # Split into batches
    batches = []
    for i in range(0, len(all_files), BATCH_SIZE):
        batches.append(all_files[i:i+BATCH_SIZE])
    
    total_batches = len(batches)
    print(f"Total batches: {total_batches}")
    
    # Process each batch
    intermediate_files = []
    for batch_num, batch_files in enumerate(batches, 1):
        output_file = os.path.join(INTERMEDIATE_DIR, f"intermediate_mosaic_{batch_num:03d}.fits")
        
        success = stitch_batch(batch_files, output_file, batch_num, total_batches)
        
        if success:
            intermediate_files.append(output_file)
        
        print(f"\nProgress: {batch_num}/{total_batches} batches completed")
    
    # Final stitching
    if len(intermediate_files) > 0:
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Successfully created {len(intermediate_files)} intermediate mosaics")
        print(f"{'='*70}")
        
        stitch_final_mosaic(intermediate_files, FINAL_OUTPUT)
        
        overall_elapsed = datetime.now() - overall_start
        print(f"\n{'='*70}")
        print(f"ALL DONE!")
        print(f"Total time: {overall_elapsed}")
        print(f"Final output: {FINAL_OUTPUT}")
        print(f"{'='*70}\n")
    else:
        print("\nERROR: No intermediate mosaics were created successfully.")

if __name__ == "__main__":
    main()