import csv
import os
import hashlib
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from tqdm import tqdm  # Add tqdm for the progress bar

logging.basicConfig(level=logging.INFO)

# List of ISIMIP priority and secondary models
isimip_models = [
    "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL",
    "CNRM-CM6-1", "CNRM-ESM2-1", "CanESM5", "EC-Earth3", "MIROC6"
]

def verify_md5(file_path, expected_md5):
    """Verify the MD5 checksum of a file."""
    if not os.path.exists(file_path):
        return False
    actual_md5 = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    return actual_md5 == expected_md5

def process_file(md5, uri):
    """Download and process the file."""
    prsout = urlparse(uri)
    ofile = os.path.split(prsout.path)[1]

    # Filter for `tas` (excluding `tasmin` and `tasmax`) or `pr`
    if ('tas' in ofile and 'tasmin' not in ofile and 'tasmax' not in ofile) or 'pr' in ofile:
        base_name = ofile.rsplit('_v1.2', 1)[0]  # Remove `_v1.2` suffix for comparison
        v1_2_file = f"{base_name}_v1.2.nc"
        yearly_mean_file = ofile.replace('.nc', '_yearmean.nc')

        # Check if the yearly mean file already exists
        if os.path.exists(yearly_mean_file):
            logging.info("Yearly mean file already exists: %s", yearly_mean_file)
            return

        # Check if the v1.2 version of the file exists
        if os.path.exists(v1_2_file):
            logging.info("File already exists (v1.2): %s", v1_2_file)
            return

        # Check if the original file exists
        if os.path.exists(ofile):
            logging.info("File already exists: %s", ofile)
            return

        logging.info("Downloading: %s", ofile)
        try:
            subprocess.run(['curl', '-s', '-o', ofile, uri], check=True)
            if not verify_md5(ofile, md5):
                logging.warning("MD5 mismatch for %s. Deleting corrupted file.", ofile)
                os.remove(ofile)
                return
            
            logging.info("Successfully downloaded: %s", ofile)

            # Compute yearly mean using CDO
            logging.info("Computing yearly mean for: %s", ofile)
            cdo_cmd = ['cdo', 'yearmean', ofile, yearly_mean_file]
            subprocess.run(cdo_cmd, check=True)
            logging.info("Yearly mean saved: %s", yearly_mean_file)

            # Remove original file
            os.remove(ofile)
            logging.info("Original file removed: %s", ofile)

        except Exception as e:
            logging.error("Error processing %s: %s", ofile, e)


# Control whether to filter for ISIMIP models or process all models
filter_isimip = True  # Set this to False to process all models

# Read CSV and process files
with open('gddp-cmip6-thredds-fileserver.csv') as index:
    fobjects = csv.reader(index)
    next(fobjects)  # Skip header

    if filter_isimip:
        # Filter tasks for files belonging to the selected ISIMIP models
        tasks = [
            (md5.strip(), uri.strip()) 
            for md5, uri in fobjects 
            if any(model in uri for model in isimip_models)
        ]
    else:
        # Include all tasks
        tasks = [(md5.strip(), uri.strip()) for md5, uri in fobjects]

# Parallel downloads and processing with progress bar
max_workers = 10
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    with tqdm(total=len(tasks), desc="Processing files") as progress_bar:
        future_to_file = {executor.submit(process_file, md5, uri): uri for md5, uri in tasks}
        for future in as_completed(future_to_file):
            uri = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                logging.error("Error processing %s: %s", uri, e)
            finally:
                progress_bar.update(1)  # Update the progress bar
