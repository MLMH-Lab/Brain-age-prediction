"""Script to run SVM on bootstrap datasets of UK BIOBANK Scanner1 using svm script"""

from pathlib import Path
import os
import glob

from src.BIOBANK.Scanner1.svm import run_svm

def main():
    # Loop over the 50 hdf5 datasets created in script create_h5_bootstrap
    for file_path in glob.iglob(
            '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/h5_datasets/homogeneous_bootstrap_*.csv',
            recursive=True):
        file_name = os.path.basename(file_path)
        print(file_name)
        run_svm(input_dataset=file_name,
                output_dir='/home/lea/PycharmProjects/predicted_brain_age/outputs/bootstrap/svm/file_name')


if __name__ == "__main__":
    main()