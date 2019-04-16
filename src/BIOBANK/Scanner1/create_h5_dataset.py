"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""

from pathlib import Path
import pandas as pd
import os

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')
# BOOTSTRAP_DIR = Path(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'bootstrap')


def create_dataset(dataset_homogeneous='homogeneous_dataset.csv',
                   input_dir='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/',
                   output_dir='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/'):

    # Load Freesurfer data as csv
    dataset_freesurfer = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')

    # Load IDs for subjects from balanced dataset
    ids_homogeneous = pd.read_csv(input_dir + dataset_homogeneous)

    # Reduce Freesurfer dataset to include only subjects from ids_homogeneous
    dataset_balanced = pd.merge(dataset_freesurfer, ids_homogeneous, on='Image_ID')

    # Loading demographic data to access age and gender per participant
    dataset_dem = pd.read_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK'/ 'Scanner1' / 'ukb22321.csv'),
        usecols=['eid', '31-0.0', '21003-2.0'])
    dataset_dem.columns = ['ID', 'Gender', 'Age']

    # Create a new col in Freesurfer dataset to contain participant ID
    dataset_balanced['Participant_ID'] = dataset_balanced['Image_ID']. \
        str.split('_', expand=True)[0]
    dataset_balanced['ID'] = dataset_balanced['Participant_ID']. \
        str.split('-', expand=True)[1]
    dataset_balanced['ID'] = pd.to_numeric(dataset_balanced['ID'])

    # Merge FS dataset and demographic dataset to access age
    dataset_csv = pd.merge(dataset_dem, dataset_balanced, on='ID')

    # Create dataset as hdf5
    dataset_name = dataset_homogeneous.split('.')[0]
    file_name = dataset_name + '_freesurferData.h5' # make sure this changed name is reflected in other scripts
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_csv.to_hdf((output_dir + file_name), key='table', mode='w')


if __name__ == "__main__":
    create_dataset()
