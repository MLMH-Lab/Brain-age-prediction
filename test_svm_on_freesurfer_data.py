"""Script to test SVM models developed using FreeSurfer data from Biobank Scanner1
on previously unseen data from Biobank Scanner2 to predict brain age.

The script loops over the 100 SVM models created in train_svm_on_freesurfer_data.py, loads their regressors,
applies them to the Scanner2 data and saves all predictions per subjects in a csv file"""

from pathlib import Path
from sklearn.externals.joblib import load

import numpy as np
import pandas as pd
import random

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    training_experiment_name = 'biobank_scanner1'
    testing_experiment_name = 'biobank_scanner2'

    # TODO: Scanner2 data still need to be cleaned and converted to h5 format
    testing_dataset_path = PROJECT_ROOT / 'outputs' / testing_experiment_name / 'freesurferData.h5'
    # ----------------------------------------------------------------------------------------
    training_experiment_dir = PROJECT_ROOT / 'outputs' / training_experiment_name
    svm_cv_dir = training_experiment_dir / 'SVM' / 'cv'

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Load hdf5 file with testing data
    testing_dataset = pd.read_hdf(testing_dataset_path, key='table')

    # Normalise regional volumes in testing dataset by total intracranial volume (tiv)
    regions = testing_dataset[COLUMNS_NAME].values

    tiv = testing_dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = testing_dataset['Age'].values

    # Create dataframe to hold actual and predicted ages in testing dataset
    testset_age_predictions = pd.DataFrame(testing_dataset[['Participant_ID', 'Age']])
    testset_age_predictions = testset_age_predictions.set_index('Participant_ID')

    # Create list of SVM model names
    n_repetitions = 10
    n_folds = 10

    regressor_list = []
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            regressor_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            regressor_list.append(regressor_filename)

    # Loop over SVM models
    for i_model in regressor_list:
        # Load regressor per model
        regressor = load(svm_cv_dir / i_model)

    # Apply regressors to new data
    # Save prediction per model in df

    # Export df as csv

if __name__ == "__main__":
    main()
