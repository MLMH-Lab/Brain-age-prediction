"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')
BOOTSTRAP_DIR = Path(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'bootstrap')


def create_dataset(dataset_homogeneous='homogeneous_dataset.csv'):
    # Load freesurfer data as csv
    dataset_freesurfer = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')

    # Load IDs for subjects from balanced dataset
    ids_homogeneous = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / dataset_homogeneous)

    # Make freesurfer dataset homogeneous
    dataset_balanced = pd.merge(dataset_freesurfer, ids_homogeneous, on='Image_ID')

    # Loading demographic data to access age per participant
    dataset_dem = pd.read_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK'/ 'Scanner1' / 'ukb22321.csv'),
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
    if dataset_homogeneous != 'homogeneous_dataset.csv':
        file_name = dataset_homogeneous + 'freesurferData.h5'
        dataset_csv.to_hdf(BOOTSTRAP_DIR / file_name,
                           key='table', mode='w')
    else:
        file_name = 'freesurferData.h5'
        dataset_csv.to_hdf(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / file_name,
                           key='table', mode='w')


if __name__ == "__main__":
    create_dataset()
