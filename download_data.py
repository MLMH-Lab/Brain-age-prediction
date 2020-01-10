#!/usr/bin/env python3
"""Script used to download the ANTs data from the storage server.

Script to download all the participants.tsv, freesurferData.csv,
 and quality metrics into the data folder.

NOTE: Only for internal use at the Machine Learning in Mental Health Lab.
"""
import argparse
from shutil import copyfile
from pathlib import Path

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--nas_path',
                    dest='nas_path_str',
                    help='Path to the Network Attached Storage system.')
args = parser.parse_args()


def download_files(data_dir, selected_path, dataset_prefix_path, nas_path):
    """Download the files necessary for the study.

    Function download files from network-attached storage.
    These files include:
        - participants.tsv: Demographic data
        - freesurferData.csv: Neuroimaging data
        - group_T1w.tsv and mriqc_prob.csv: Raw data quality metrics
        - qoala_prob.csv: Freesurfer data quality metrics

    Parameters
    ----------
    data_dir: PosixPath
        Path indicating local path to store the data.
    selected_path: PosixPath
        Path indicating external path with the data.
    dataset_prefix_path: str
        Datasets prefix.
    nas_path: PosixPath
        Path indicating NAS system.
    """

    dataset_path = data_dir / dataset_prefix_path
    dataset_path.mkdir(exist_ok=True, parents=True)

    copyfile(str(selected_path / 'participants.tsv'), str(dataset_path / 'participants.tsv'))

    try:
        copyfile(str(nas_path / 'FreeSurfer_preprocessed' / dataset_prefix_path / 'freesurferData.csv'),
                 str(dataset_path / 'freesurferData.csv'))
    except:
        print('{} does not have freesurferData.csv'.format(dataset_prefix_path))

    try:
        copyfile(str(nas_path / 'MRIQC' / dataset_prefix_path / 'group_T1w.tsv'),
                 str(dataset_path / 'group_T1w.tsv'))
    except:
        print('{} does not have group_T1w.tsv'.format(dataset_prefix_path))

    try:
        mriqc_prob_path = next((nas_path / 'MRIQC' / dataset_prefix_path).glob('*unseen_pred.csv'))
        copyfile(str(mriqc_prob_path), str(dataset_path / 'mriqc_prob.csv'))
    except:
        print('{} does not have *unseen_pred.csv'.format(dataset_prefix_path))

    try:
        qoala_prob_path = next((nas_path / 'Qoala' / dataset_prefix_path).glob('Qoala*'))
        copyfile(str(qoala_prob_path), str(dataset_path / 'qoala_prob.csv'))
    except:
        print('{} does not have Qoala*'.format(dataset_prefix_path))


def main(nas_path_str):
    """Perform download of selected datasets from the network-attached storage."""
    nas_path = Path(nas_path_str)
    data_dir = PROJECT_ROOT / 'data'

    dataset_name = 'BIOBANK'
    selected_path = nas_path / 'BIDS_data' / dataset_name

    for subdirectory_selected_path in selected_path.iterdir():
        if not subdirectory_selected_path.is_dir():
            continue

        print(subdirectory_selected_path)

        scanner_name = subdirectory_selected_path.stem
        if (subdirectory_selected_path / 'participants.tsv').is_file():
            download_files(data_dir, subdirectory_selected_path, dataset_name + '/' + scanner_name, nas_path)


if __name__ == "__main__":
    main(args.nas_path_str)
