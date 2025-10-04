import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from reproject import reproject_interp
import glob
import os

# --- Configuration ---
INPUT_DIRECTORY = " " 
FITS_FILE_PATTERN = os.path.join(INPUT_DIRECTORY, '*ffic.fits')
OUTPUT_MOSAIC_FITS = "tess_ccd_mosaic_stitched.fits"

# --- Helper Function to Clean WCS ---
def get_2d_celestial_data(filename):
    """
    Reads a TESS FFI FITS file, extracts the 2D image data, 
    and returns a WCS object explicitly configured for 2D celestial coordinates.
    """
    with fits.open(filename, mode='readonly') as hdulist:
        # Try to find the HDU with valid WCS data
        # TESS calibrated FFIs usually store the image and WCS in HDU 1, but check all HDUs
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
        
        # Debug: Print what coordinate types we have
        print(f"\nDEBUG {os.path.basename(filename)}:")
        for i in range(1, 10):
            ctype_key = f'CTYPE{i}'
            if ctype_key in header:
                print(f"  {ctype_key} = {header[ctype_key]}")
        
        # 1. Handle multi-dimensional data first
        if data.ndim == 3:
            mid_cadence = data.shape[0] // 2
            data_2d = data[mid_cadence, :, :]
        elif data.ndim == 2:
            data_2d = data
        else:
            raise ValueError(f"Unexpected data dimensions: {data.ndim}")
        
        # 2. Create WCS and check what we have
        wcs_full = WCS(header)
        print(f"  Full WCS naxis: {wcs_full.naxis}")
        print(f"  WCS celestial check: {wcs_full.has_celestial}")
        
        # 3. Strategy: Try different approaches to get celestial WCS
        wcs_2d = None
        
        # Approach 1: Use .celestial if it exists and has celestial components
        if wcs_full.has_celestial:
            try:
                wcs_2d = wcs_full.celestial
                print(f"  ✓ Using .celestial property: naxis={wcs_2d.naxis}")
            except Exception as e:
                print(f"  ✗ .celestial failed: {e}")
        
        # Approach 2: If that didn't work, try sub-selecting specific axes
        if wcs_2d is None or not wcs_2d.has_celestial:
            print(f"  Attempting to find RA/Dec axes manually...")
            # Look for RA and Dec axes
            ra_axis = None
            dec_axis = None
            
            for i in range(wcs_full.naxis):
                ctype = wcs_full.wcs.ctype[i]
                if 'RA' in ctype or 'LON' in ctype:
                    ra_axis = i
                    print(f"    Found RA axis at position {i}: {ctype}")
                elif 'DEC' in ctype or 'LAT' in ctype:
                    dec_axis = i
                    print(f"    Found DEC axis at position {i}: {ctype}")
            
            if ra_axis is not None and dec_axis is not None:
                # Sub-select these specific axes
                try:
                    wcs_2d = wcs_full.sub([ra_axis + 1, dec_axis + 1])  # WCS uses 1-based indexing
                    print(f"  ✓ Created WCS from axes {ra_axis} and {dec_axis}: naxis={wcs_2d.naxis}")
                except Exception as e:
                    print(f"  ✗ Axis sub-selection failed: {e}")
        
        # Approach 3: Last resort - build from scratch
        if wcs_2d is None or not wcs_2d.has_celestial or wcs_2d.naxis != 2:
            print(f"  Building WCS from scratch using header keywords...")
            wcs_2d = WCS(naxis=2)
            
            # Find which axes in the header correspond to celestial coordinates
            # TESS typically uses 1 and 2, but let's be thorough
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
            
            print(f"    Using header axes {celestial_axis_1} and {celestial_axis_2}")
            
            # Copy WCS parameters from the identified celestial axes
            wcs_2d.wcs.ctype[0] = header.get(f'CTYPE{celestial_axis_1}', 'RA---TAN')
            wcs_2d.wcs.ctype[1] = header.get(f'CTYPE{celestial_axis_2}', 'DEC--TAN')
            wcs_2d.wcs.crval[0] = header.get(f'CRVAL{celestial_axis_1}', 0.0)
            wcs_2d.wcs.crval[1] = header.get(f'CRVAL{celestial_axis_2}', 0.0)
            wcs_2d.wcs.crpix[0] = header.get(f'CRPIX{celestial_axis_1}', 1.0)
            wcs_2d.wcs.crpix[1] = header.get(f'CRPIX{celestial_axis_2}', 1.0)
            
            # Handle CD matrix
            cd_keys = [
                (f'CD{celestial_axis_1}_{celestial_axis_1}', f'CD{celestial_axis_1}_{celestial_axis_2}'),
                (f'CD{celestial_axis_2}_{celestial_axis_1}', f'CD{celestial_axis_2}_{celestial_axis_2}')
            ]
            
            if all(key in header for row in cd_keys for key in row):
                wcs_2d.wcs.cd = [
                    [header[cd_keys[0][0]], header[cd_keys[0][1]]],
                    [header[cd_keys[1][0]], header[cd_keys[1][1]]]
                ]
                print(f"    ✓ Set CD matrix")
            elif f'CDELT{celestial_axis_1}' in header and f'CDELT{celestial_axis_2}' in header:
                wcs_2d.wcs.cdelt[0] = header[f'CDELT{celestial_axis_1}']
                wcs_2d.wcs.cdelt[1] = header[f'CDELT{celestial_axis_2}']
                print(f"    ✓ Set CDELT values")
            
            # Set additional parameters if available
            if 'RADESYS' in header:
                wcs_2d.wcs.radesys = header['RADESYS']
            if 'EQUINOX' in header:
                wcs_2d.wcs.equinox = header['EQUINOX']
        
        # Final verification
        if wcs_2d.naxis != 2:
            raise ValueError(f"WCS is not 2D (naxis={wcs_2d.naxis}) for {filename}")
        
        if not wcs_2d.has_celestial:
            raise ValueError(f"WCS does not have celestial components for {filename}")
        
        print(f"  ✓✓ Final WCS: naxis={wcs_2d.naxis}, celestial={wcs_2d.has_celestial}, ctype={wcs_2d.wcs.ctype}")

    return data_2d, wcs_2d

# --- Main Mosaicking Function ---
def stitch_tess_mosaic(file_pattern, output_filename):
    """
    Stitches multiple FITS files (FFIs) into a single FITS mosaic using WCS.
    """
    input_files = sorted(glob.glob(file_pattern))
    print(f"Found {len(input_files)} FITS files to stitch.\n")
    
    if len(input_files) < 2:
        print("ERROR: Need at least 2 FITS files to create a mosaic.")
        return

    # Load the image and WCS data as clean (data, WCS) tuples
    input_data_wcs = []
    for filename in input_files:
        try:
            data, wcs = get_2d_celestial_data(filename)
            
            # Verification before adding
            if wcs.naxis != 2:
                print(f"ERROR: Skipping {filename} - WCS is {wcs.naxis}D, not 2D")
                continue
            
            if not wcs.has_celestial:
                print(f"ERROR: Skipping {filename} - WCS lacks celestial components")
                continue
                
            input_data_wcs.append((data, wcs))
            print(f"✓ Added {os.path.basename(filename)} to mosaic list\n")
            
        except Exception as e:
            print(f"ERROR: Failed to process {filename}. Skipping. Error: {e}\n")

    if len(input_data_wcs) < 2:
        print("ERROR: Not enough valid FITS files loaded.")
        return

    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(input_data_wcs)} files for mosaicking.")
    print(f"{'='*60}\n")

    # Calculate the Optimal WCS and Output Shape
    print("Calculating optimal mosaic boundaries (WCS)...")
    
    reference_wcs, shape_out = find_optimal_celestial_wcs(
        input_data_wcs,
        frame='icrs', 
        projection='TAN'
    )
    print(f"Optimal Mosaic Size (pixels): {shape_out}")

    # Reproject and Co-add (The Stitching)
    print("\nStarting re-projection and co-addition (stitching)...")
    print("This may take several minutes...\n")
    
    mosaic_array, footprint = reproject_and_coadd(
        input_data_wcs,
        reference_wcs,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function='mean'    
    )

    # Save the Final Mosaic
    hdu = fits.PrimaryHDU(data=mosaic_array, header=reference_wcs.to_header())
    hdu.writeto(output_filename, overwrite=True)
    print(f"\nSUCCESS: Mosaic saved as: {output_filename}")

    # Display the Result
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
    plt.title('TESS FFI Mosaiced Image')
    plt.tight_layout()
    plt.savefig('tess_mosaic_preview.png', dpi=150, bbox_inches='tight')
    print("Preview saved as: tess_mosaic_preview.png")
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    stitch_tess_mosaic(FITS_FILE_PATTERN, OUTPUT_MOSAIC_FITS)