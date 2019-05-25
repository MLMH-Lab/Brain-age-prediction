"""Script to create bootstrap datasets of UK BIOBANK Scanner1
in hdf5 format using create_h5_dataset script"""


from pathlib import Path
import os
import glob

# from src.BIOBANK.Scanner1.create_h5_dataset import create_dataset
from create_h5_dataset import create_dataset # to run it from terminal

PROJECT_ROOT = Path.cwd()


def main():
    # Loop over the 50 samples created in script create_bootstrap_ids
    for file_path in glob.iglob(
            '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/bootstrap_ids/homogeneous_bootstrap_*.csv',
            recursive=True):
        file_name = os.path.basename(file_path)
        print(file_name)
        create_dataset(dataset_homogeneous=file_name,
                       input_dir='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/bootstrap_ids',
                       output_dir='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/h5_datasets')


if __name__ == "__main__":
    main()