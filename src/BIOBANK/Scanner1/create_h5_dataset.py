"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():
    # Load freesurfer data as csv
    dataset_freesurfer = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')

    # Load IDs for subjects from balanced dataset
    ids_homogeneous = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/homogeneous_dataset.csv')

    # Make freesurfer dataset homogeneous
    dataset_balanced = pd.merge(dataset_freesurfer, ids_homogeneous, on='Image_ID')

    # Loading demographic data to access age per participant
    dataset_dem = pd.read_csv(
        '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/ukb22321.csv',
        usecols=['eid', '21003-2.0'])
    dataset_dem.columns = ['ID', 'Age']
    dataset_dem = dataset_dem.dropna()

    # Create a new col in FS dataset to contain participant ID
    dataset_balanced['Participant_ID'] = dataset_balanced['Image_ID']. \
        str.split('_', expand=True)[0]
    dataset_balanced['ID'] = dataset_balanced['Participant_ID']. \
        str.split('-', expand=True)[1]
    dataset_balanced['ID'] = pd.to_numeric(dataset_balanced['ID'])

    # Merge FS dataset and demographic dataset to access age
    dataset_csv = pd.merge(dataset_dem, dataset_balanced, on='ID')

    # Create dataset as hdf5
    dataset_csv.to_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table', mode='w')


if __name__ == "__main__":
    main()
