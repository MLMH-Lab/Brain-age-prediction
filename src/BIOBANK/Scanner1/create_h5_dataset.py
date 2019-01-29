"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():

    # Load freesurfer data as csv
    dataset_freesurfer = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')

    # Loading demographic data to access age per participant
    dataset_demographic = pd.read_csv('/home/lea/PycharmProjects/'
                                      'predicted_brain_age/data/BIOBANK/Scanner1/participants.tsv', sep='\t')
    dataset_demographic_excl_nan = dataset_demographic.dropna()

    # Create a new col in FS dataset to contain Participant_ID
    dataset_freesurfer['Participant_ID'] = dataset_freesurfer['Image_ID']. \
        str.split('_', expand=True)[0]

    # Merge FS dataset and demographic dataset to access age
    dataset_csv = pd.merge(dataset_demographic_excl_nan[['Participant_ID', 'Age']], dataset_freesurfer, on='Participant_ID')

    # Create dataset as hdf5
    dataset_hdf = dataset_csv.to_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table', mode='w')


if __name__ == "__main__":
    main()