"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()


def main():
    # Load freesurfer data
    dataset_fs = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv')

    # Load IDs for subjects from balanced dataset
    ids_homogeneous = pd.read_csv(PROJECT_ROOT / 'outputs' / 'homogeneous_dataset.csv')

    # Make freesurfer dataset homogeneous
    dataset_balanced = pd.merge(dataset_fs, ids_homogeneous, on='Image_ID')

    # Loading demographic data to access age per participant
    dataset_dem = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv', sep='\t')
    dataset_dem = dataset_dem.dropna()

    # Create a new col in FS dataset to contain participant ID
    dataset_balanced['Participant_ID'] = dataset_balanced['Image_ID'].str.split('_', expand=True)[0]

    # Merge FS dataset and demographic dataset to access age
    dataset_csv = pd.merge(dataset_dem, dataset_balanced, on='Participant_ID')

    # Create datasets as hdf5
    dataset_csv.to_hdf(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData_total.h5', key='table',
                       mode='w')


if __name__ == "__main__":
    main()
