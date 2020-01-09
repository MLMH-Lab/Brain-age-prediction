"""Script to test SVM models developed using FreeSurfer data from Biobank Scanner1
on previously unseen data from Biobank Scanner2 to predict brain age.

The script loops over the 100 SVM models created in train_svm_on_freesurfer_data.py, loads their regressors,
applies them to the Scanner2 data and saves all predictions per subjects in a csv file"""
from math import sqrt
from pathlib import Path
from sklearn.externals.joblib import load

from scipy import stats
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    training_experiment_name = 'biobank_scanner1'
    testing_experiment_name = 'biobank_scanner2'

    model_name = 'RVM'
    testing_dataset_path = PROJECT_ROOT / 'outputs' / testing_experiment_name / 'freesurferData.h5'
    # ----------------------------------------------------------------------------------------
    training_experiment_dir = PROJECT_ROOT / 'outputs' / training_experiment_name
    svm_cv_dir = training_experiment_dir / model_name / 'cv'
    test_cv_dir = PROJECT_ROOT / 'outputs' / testing_experiment_name / model_name / 'cv'
    test_cv_dir.mkdir(parents=True)

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Load hdf5 file with testing data
    dataset = pd.read_hdf(testing_dataset_path, key='table')

    # Normalise regional volumes in testing dataset by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    # Create dataframe to hold actual and predicted ages
    age_predictions = pd.DataFrame(dataset[['Participant_ID', 'Age']])
    age_predictions = age_predictions.set_index('Participant_ID')

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
            x_test = scaler.transform(regions_norm)

            # Apply regressors to scaled data
            predictions = regressor.predict(x_test)

            absolute_error = mean_absolute_error(age, predictions)
            root_squared_error = sqrt(mean_squared_error(age, predictions))
            r2_score = regressor.score(x_test, age)
            age_error_corr, _ = stats.spearmanr(np.abs(age - predictions), age)

            # Save prediction per model in df
            age_predictions[('{:02d}_{:02d}'.format(i_repetition, i_fold))] = predictions

            # Save model scores r2, MAE, RMSE
            scores_array = np.array([r2_score, absolute_error, root_squared_error, age_error_corr])
            scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)
            np.save(test_cv_dir / scores_filename, scores_array)

    # Export df as csv
    testset_age_predictions_filename = PROJECT_ROOT / 'outputs' / testing_experiment_name / 'svm_testset_predictions.csv'
    age_predictions.to_csv(testset_age_predictions_filename)


if __name__ == "__main__":
    main()
