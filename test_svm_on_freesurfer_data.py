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

    # Create dataframe to hold actual and predicted ages in testing dataset
    testset_age_predictions = pd.DataFrame(testing_dataset[['Participant_ID', 'Age']])
    testset_age_predictions = testset_age_predictions.set_index('Participant_ID')

    # Create list of SVM model prefixes
    n_repetitions = 10
    n_folds = 10

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            # Load regressor, scaler and parameters per model
            regressor_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            regressor = load(svm_cv_dir / regressor_filename)

            scaler_filename = '{:02d}_{:02d}_scaler.joblib'.format(i_repetition, i_fold)
            scaler = load(svm_cv_dir / scaler_filename)

            # Use RobustScaler to transform testing data
            regions_norm_test = scaler.transform(regions_norm)

            # Apply regressors to scaled data
            test_predictions = regressor.predict(regions_norm_test)

            # Save prediction per model in df
            testset_age_predictions[('{:02d}_{:02d}'.format(i_repetition, i_fold))] = test_predictions

    # Export df as csv
    testset_age_predictions_filename = PROJECT_ROOT / 'outputs' / testing_experiment_name / 'svm_testset_predictions.csv'
    testset_age_predictions.to_csv(testset_age_predictions_filename)

if __name__ == "__main__":
    main()
