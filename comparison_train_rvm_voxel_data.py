#!/usr/bin/env python3
"""Script to train Relevant Vector Machines on voxel data.

We trained the Relevant Vector Machines (RVMs) [1] in a 10 repetitions
10 stratified k-fold cross-validation (stratified by age).
The hyperparameter tuning was performed in an automatic way using
 a nested cross-validation.
This script assumes that a kernel has been already pre-computed.
To compute the kernel use the script `precompute_3Ddata.py`

References
----------
[1] - Tipping, Michael E. "The relevance vector machine."
Advances in neural information processing systems. 2000.
"""
import argparse
from math import sqrt
from pathlib import Path
import random
import warnings
import sys

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn_rvm import EMRVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals.joblib import dump

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')
parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')
parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='homogenized_ids.csv',
                    help='Filename indicating the ids to be used.')
args = parser.parse_args()


def main(experiment_name, scanner_name, input_ids_file):
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    ids_path = experiment_dir / input_ids_file

    model_dir = experiment_dir / 'voxel_RVM'
    model_dir.mkdir(exist_ok=True)
    cv_dir = model_dir / 'cv'
    cv_dir.mkdir(exist_ok=True)

    # Load demographics
    demographics = load_demographic_data(participants_path, ids_path)

    # Load the Gram matrix
    kernel_path = PROJECT_ROOT / 'outputs' / 'kernels' / 'kernel.csv'
    kernel = pd.read_csv(kernel_path, header=0, index_col=0)

    # ----------------------------------------------------------------------------------------
    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    age = demographics['Age'].values

    # Cross validation variables
    cv_r2_scores = []
    cv_mae = []
    cv_rmse = []
    cv_age_error_corr = []

    # Create Dataframe to hold actual and predicted ages
    age_predictions = demographics[['Image_ID', 'Age']]
    age_predictions = age_predictions.set_index('Image_ID')

    n_repetitions = 10
    n_folds = 10

    for i_repetition in range(n_repetitions):
        # Create new empty column in age_predictions df to save age predictions of this repetition
        repetition_column_name = 'Prediction repetition {:02d}'.format(i_repetition)
        age_predictions[repetition_column_name] = np.nan

        # Create 10-fold cross-validation scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(kernel, age)):
            print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))

            x_train = kernel.iloc[train_index, train_index].values
            x_test = kernel.iloc[test_index, train_index].values
            y_train, y_test = age[train_index], age[test_index]

            model = EMRVR(kernel='precomputed')

            model.fit(x_train, y_train)

            predictions = model.predict(x_test)

            absolute_error = mean_absolute_error(y_test, predictions)
            root_squared_error = sqrt(mean_squared_error(y_test, predictions))
            r2_score = model.score(x_test, y_test)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

            cv_r2_scores.append(r2_score)
            cv_mae.append(absolute_error)
            cv_rmse.append(root_squared_error)
            cv_age_error_corr.append(age_error_corr)

            # Save scaler, model and model parameters
            model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            params_filename = '{:02d}_{:02d}_params.joblib'.format(i_repetition, i_fold)

            # Save model scores r2, MAE, RMSE
            scores_array = np.array([r2_score, absolute_error, root_squared_error, age_error_corr])

            scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)

            np.save(cv_dir / scores_filename, scores_array)

            # Add predictions per test_index to age_predictions
            for row, value in zip(test_index, predictions):
                age_predictions.iloc[row, age_predictions.columns.get_loc(repetition_column_name)] = value

            # Print results of the CV fold
            print('Repetition {:02d}, Fold {:02d}, R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}, CORR: {:0.3f}'
                  .format(i_repetition, i_fold, r2_score, absolute_error, root_squared_error, age_error_corr))

    # Save predictions
    age_predictions.to_csv(model_dir / 'age_predictions.csv')

    # Variables for CV means across all repetitions
    cv_r2_mean = np.mean(cv_r2_scores)
    cv_mae_mean = np.mean(cv_mae)
    cv_rmse_mean = np.mean(cv_rmse)
    cv_age_error_corr_mean = np.mean(np.abs(cv_age_error_corr))
    print('Mean R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}, CORR: {:0.3f}'.format(cv_r2_mean,
                                                                                cv_mae_mean,
                                                                                cv_rmse_mean,
                                                                                cv_age_error_corr_mean))


if __name__ == "__main__":
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
