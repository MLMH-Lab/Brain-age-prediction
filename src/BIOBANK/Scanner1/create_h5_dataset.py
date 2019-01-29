"""Script to create dataset of UK BIOBANK Scanner1 in hdf5 format"""


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():

    # Load freesurfer data as hdf5
    dataset_csv = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')
    dataset_hdf = dataset_csv.to_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table', mode='w')


if __name__ == "__main__":
    main()