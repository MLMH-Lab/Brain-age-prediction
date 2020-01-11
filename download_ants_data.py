#!/usr/bin/env python3
"""Script used to download the ANTs data from the storage server.

Script to download all the UK BIOBANK files preprocessed using the
scripts available at the imaging_preprocessing_ANTs folder.

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

parser.add_argument('-O', '--output_path',
                    dest='output_path_str',
                    help='Path to the local output folder.')

args = parser.parse_args()


def main(nas_path_str, output_path_str):
    """Perform download of selected datasets from the network-attached storage."""
    nas_path = Path(nas_path_str)
    output_path = Path(output_path_str)

    dataset_name = 'BIOBANK'
    scanner_name = 'SCANNER01'

    dataset_output_path = output_path / dataset_name
    dataset_output_path.mkdir(exist_ok=True)

    selected_path = nas_path / 'ANTS_NonLinear_preprocessed' / dataset_name / scanner_name

    for file_path in selected_path.glob('*.nii.gz'):
        print(file_path)
        copyfile(str(file_path), str(dataset_output_path / file_path.name))


if __name__ == '__main__':
    main(args.nas_path_str, args.output_path_str)
