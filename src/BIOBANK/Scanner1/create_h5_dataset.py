"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():

    # Load freesurfer data as csv
    dataset_freesurfer = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')

    # Loading demographic data to access age per participant
    dataset_demographic = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/homogeneous_dataset.csv')

    # Create a new col in FS dataset to contain participant ID
    dataset_freesurfer['ID'] = dataset_freesurfer['Image_ID']. \
        str.split('_', expand=True)[0]

    # Merge FS dataset and demographic dataset to access age
    dataset_csv = pd.merge(dataset_demographic[['ID', 'Age']], dataset_freesurfer, on='ID')

    # Create dataset as hdf5
    dataset_hdf = dataset_csv.to_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table', mode='w')


if __name__ == "__main__":
    main()