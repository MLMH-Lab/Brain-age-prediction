#!/usr/bin/env python3
"""Script used to download the study data from the storage server.

Script to download all the participants.tsv and freesurferData.csv into the data folder.

NOTE: Only for internal use at the Machine Learning in Mental Health Lab.
"""
import argparse
from shutil import copyfile
from pathlib import Path

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()
parser.add_argument('-P', '--path_nas',
                    dest='path_nas',
                    help='Path to the Network Attached Storage system.')
args = parser.parse_args()


def download_files(data_dir, selected_path, dataset_prefix_path, path_nas):
    """Download the files necessary for the study.

    Function download files from network-attached storage.
    These files include:
        - participants.tsv: Demographic data
        - freesurferData.csv: Neuroimaging data

    Parameters
    ----------
    data_dir: PosixPath
        Path indicating local path to store the data.
    selected_path: PosixPath
        Path indicating external path with the data.
    dataset_prefix_path: str
        Datasets prefix.
    path_nas: PosixPath
        Path indicating NAS system.
    """

    dataset_path = data_dir / dataset_prefix_path
    dataset_path.mkdir(exist_ok=True, parents=True)

    try:
        copyfile(str(selected_path / 'participants.tsv'), str(dataset_path / 'participants.tsv'))
        copyfile(str(path_nas / 'FreeSurfer_preprocessed' / dataset_prefix_path / 'freesurferData.csv'),
                 str(dataset_path / 'freesurferData.csv'))
    except:
        print('{} does not have freesurferData.csv'.format(dataset_prefix_path))


def main(path_nas):
    """Perform download of selected datasets from the network-attached storage."""
    path_nas = Path(path_nas)
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)

    dataset_name = 'BIOBANK'
    selected_path = path_nas / 'BIDS_data' / dataset_name

    for subdirectory_selected_path in selected_path.iterdir():
        if not subdirectory_selected_path.is_dir():
            continue

        print(subdirectory_selected_path)

        scanner_name = subdirectory_selected_path.stem
        if (subdirectory_selected_path / 'participants.tsv').is_file():
            download_files(data_dir, subdirectory_selected_path, dataset_name + '/' + scanner_name, path_nas)


if __name__ == "__main__":
    main(args.path_nas)
