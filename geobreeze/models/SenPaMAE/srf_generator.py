import numpy as np
import pandas as pd
import yaml
import math
import os

def gaussian(x, mu, sigma):
    """
    Gaussian function
    
    Parameters:
    -----------
    x : float or array
        Input value(s)
    mu : float
        Mean (center) of the gaussian
    sigma : float
        Standard deviation of the gaussian
        
    Returns:
    --------
    float or array
        Gaussian values for input x
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_srf_dataframe(yaml_file, wavelength_min=300, wavelength_max=2500, wavelength_step=1):
    """
    Generate a dataframe with spectral response functions for all bands
    
    Parameters:
    -----------
    yaml_file : str
        Path to YAML file with band specifications
    wavelength_min : int
        Minimum wavelength in nm
    wavelength_max : int
        Maximum wavelength in nm
    wavelength_step : int
        Wavelength step size in nm
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with wavelengths as index and bands as columns
    """
    # Read the YAML file
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create wavelength range (inclusive of both min and max)
    wavelengths = np.arange(wavelength_min, wavelength_max + 1, wavelength_step)
    
    # Initialize empty dataframe
    srf_df = pd.DataFrame(index=wavelengths)
    srf_df.index.name = 'wavelength'
    
    # Process each band
    bands = config['bands']
    for band_name, band_info in bands.items():
        if 'gaussian' in band_info:
            mu = band_info['gaussian']['mu']
            sigma = band_info['gaussian']['sigma']
            
            # Calculate gaussian response for all wavelengths
            response = gaussian(wavelengths, mu, sigma)
            
            # Normalize so peak is at 1.0
            peak_response = gaussian(mu, mu, sigma)
            response = response / peak_response
            
            # Add to dataframe
            srf_df[band_name] = response
    
    return srf_df

def main():
    """
    Main function to generate and save the SRF dataframe
    """
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate spectral response functions from sensor specifications.')
    parser.add_argument('input_file', type=str, help='Input YAML file with sensor band specifications')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output CSV file name (default: input_filename_srf.csv)')
    parser.add_argument('--min-wavelength', type=int, default=300, 
                        help='Minimum wavelength in nm (default: 300)')
    parser.add_argument('--max-wavelength', type=int, default=2600, 
                        help='Maximum wavelength in nm (default: 2600)')
    parser.add_argument('--step', type=int, default=1, 
                        help='Wavelength step size in nm (default: 1)')
    
    args = parser.parse_args()
    
    # Set input and output files
    yaml_file = args.input_file
    
    # If output file not specified, derive from input file name
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(yaml_file))[0]
        output_file = f"{base_name}_srf.csv"
        output_npy = f"{base_name}_srf.npy"
    else:
        output_file = args.output
        # Create .npy filename from the output file
        output_npy = os.path.splitext(output_file)[0] + '.npy'
    
    # Generate SRF dataframe
    print(f"Generating SRF from {yaml_file}...")
    srf_df = generate_srf_dataframe(yaml_file, 
                                   wavelength_min=args.min_wavelength,
                                   wavelength_max=args.max_wavelength,
                                   wavelength_step=args.step)
    
    # Save to CSV
    srf_df.to_csv(output_file)
    print(f"SRF saved to {output_file}")
    
    # Save to NPY
    # Transpose the dataframe to get [wavelength, bands] format
    import numpy as np
    # Extract wavelengths as first column and band data as the rest
    wavelengths = srf_df.index.values
    band_data = srf_df.values
    
    # Create a 2D array with shape [wavelength, bands]
    srf_array = band_data
    
    # Save the array to a .npy file
    np.save(output_npy, srf_array)
    print(f"SRF saved to {output_npy} with shape {srf_array.shape}")
    
    # Print some basic statistics
    print(f"Number of bands: {len(srf_df.columns)}")
    print(f"Wavelength range: {srf_df.index.min()} - {srf_df.index.max()} nm")
    print(f"First 5 bands: {', '.join(srf_df.columns[:5])}")

if __name__ == "__main__":
    main()