#!/usr/bin/env python3
"""Script to train Relevant Vector Machines on voxel-level data.

We trained the Relevant Vector Machines (RVMs) [1] in 10 repetitions of
10 stratified k-fold cross-validation (CV) (stratified by age).

This script assumes that a kernel has been already pre-computed.
To compute the kernel use the script `precompute_3Ddata.py`

References
----------
[1] - Tipping, Michael E. "The relevance vector machine."
Advances in neural information processing systems. 2000.
"""
import argparse
import random
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn_rvm import EMRVR

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
                    help='File name indicating the subject IDs to be used.')

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

    # CV variables
    cv_r2 = []
    cv_mae = []
    cv_rmse = []
    cv_age_error_corr = []

    # Create DataFrame to hold actual and predicted ages
    age_predictions = demographics[['image_id', 'Age']]
    age_predictions = age_predictions.set_index('image_id')

    n_repetitions = 10
    n_folds = 10

    for i_repetition in range(n_repetitions):
        # Create new empty column in age_predictions df to save age predictions of this repetition
        repetition_column_name = f'Prediction repetition {i_repetition:02d}'
        age_predictions[repetition_column_name] = np.nan

        # Create 10-fold CV scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(kernel, age)):
            print(f'Running repetition {i_repetition:02d}, fold {i_fold:02d}')

            x_train = kernel.iloc[train_index, train_index].values
            x_test = kernel.iloc[test_index, train_index].values
            y_train, y_test = age[train_index], age[test_index]

            model = EMRVR(kernel='precomputed', threshold_alpha=1e9)

            model.fit(x_train, y_train)

            predictions = model.predict(x_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

            cv_r2.append(r2)
            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_age_error_corr.append(age_error_corr)

            # ----------------------------------------------------------------------------------------
            # Save output files
            output_prefix = f'{i_repetition:02d}_{i_fold:02d}'

            # Save model
            dump(model, cv_dir / f'{output_prefix}_regressor.joblib')

            # Save model scores
            scores_array = np.array([r2, mae, rmse, age_error_corr])
            np.save(cv_dir / f'{output_prefix}_scores.npy', scores_array)

            # Save train index
            np.save(cv_dir / f'{output_prefix}_train_index.npy', train_index)

            # ----------------------------------------------------------------------------------------
            # Add predictions per test_index to age_predictions
            for row, value in zip(test_index, predictions):
                age_predictions.iloc[row, age_predictions.columns.get_loc(repetition_column_name)] = value

            # Print results of the CV fold
            print(f'Repetition {i_repetition:02d} Fold {i_fold:02d} R2: {r2:0.3f}, '
                  f'MAE: {mae:0.3f} RMSE: {rmse:0.3f} CORR: {age_error_corr:0.3f}')

    # Save predictions
    age_predictions.to_csv(model_dir / 'age_predictions.csv')

    # Variables for mean scores of performance metrics of CV folds across all repetitions
    print('')
    print('Mean values:')
    print(f'R2: {np.mean(cv_r2):0.3f} MAE: {np.mean(cv_mae):0.3f} '
          f'RMSE: {np.mean(cv_rmse):0.3f} CORR: {np.mean(np.abs(cv_age_error_corr)):0.3f}')


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
