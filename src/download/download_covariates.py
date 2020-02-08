#!/usr/bin/env python3
"""Script used to download the ANTs data from the storage server.

Script to download all the participants.tsv, freesurferData.csv,
 and quality metrics into the data folder.

NOTE: Only for internal use at the Machine Learning in Mental Health Lab.
"""
import argparse
from pathlib import Path
from shutil import copyfile

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-N', '--nas_path',
                    dest='nas_path_str',
                    help='Path to the Network Attached Storage system.')

args = parser.parse_args()


def main(nas_path_str):
    """Perform download of selected datasets from the network-attached storage."""
    nas_path = Path(nas_path_str)
    data_dir = PROJECT_ROOT / 'data'

    dataset_name = 'BIOBANK'
    selected_path = nas_path / 'original_data' / dataset_name / 'Original_Files'

    copyfile(str(selected_path / 'ukb22321.csv'), str(data_dir / dataset_name / 'ukb22321.csv'))


if __name__ == '__main__':
    main(args.nas_path_str)
