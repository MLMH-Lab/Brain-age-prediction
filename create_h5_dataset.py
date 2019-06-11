"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""

from pathlib import Path

import pandas as pd

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def create_dataset(demographic_path, id_path, freesurfer_path, output_dir):
    """Perform the exploratory data analysis."""
    dataset = load_demographic_data(demographic_path, id_path)

    # Loading Freesurfer data
    freesurfer = pd.read_csv(freesurfer_path)

    # Create a new col in FS dataset to contain Participant_ID
    freesurfer['Participant_ID'] = freesurfer['Image_ID'].str.split('_', expand=True)[0]

    # Merge FS dataset and demographic dataset to access age
    dataset = pd.merge(freesurfer, dataset, on='Participant_ID')

    # Create dataset as hdf5
    dataset.to_hdf(output_dir / 'freesurferData.h5', key='table', mode='w')


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'dataset_homogeneous.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'
    output_dir = PROJECT_ROOT / 'outputs' / experiment_name
    # ----------------------------------------------------------------------------------------

    create_dataset(demographic_path, id_path, freesurfer_path, output_dir)
