#!/usr/bin/env python3
"""Perform sample size Script to run SVM (linear SVR) on bootstrap datasets of UK BIOBANK Scanner1
IMPORTANT NOTE: This script is adapted from svm.py but uses KFold instead of StratifiedKFold
to account for the bootstrap samples with few participants
"""
import argparse
import random
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR

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

parser.add_argument('-N', '--n_bootstrap',
                    dest='n_bootstrap',
                    type=int, default=1000,
                    help='Number of bootstrap iterations.')

parser.add_argument('-R', '--n_max_pair',
                    dest='n_max_pair',
                    type=int, default=20,
                    help='Number maximum of pairs.')

args = parser.parse_args()


def main(experiment_name, scanner_name, n_bootstrap, n_max_pair):
    # ----------------------------------------------------------------------------------------
    model_name = 'voxel_SVM'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'

    # Load the Gram matrix
    kernel_path = PROJECT_ROOT / 'outputs' / 'kernels' / 'kernel.csv'
    kernel = pd.read_csv(kernel_path, header=0, index_col=0)

    # ----------------------------------------------------------------------------------------
    # Loop over the 20 bootstrap samples with up to 20 gender-balanced subject pairs per age group/year
    for i_n_subject_pairs in range(1, n_max_pair + 1):
        print(f'Bootstrap number of subject pairs: {i_n_subject_pairs}')
        ids_with_n_subject_pairs_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}' / 'ids'

        scores_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}' / 'scores'
        scores_dir.mkdir(exist_ok=True)

        # Loop over the 1000 random subject samples per bootstrap
        for i_bootstrap in range(n_bootstrap):
            print(f'Sample number within bootstrap: {i_bootstrap}')

            prefix = f'{i_bootstrap:04d}_{i_n_subject_pairs:02d}'
            train_dataset = load_demographic_data(participants_path,
                                                  ids_with_n_subject_pairs_dir / f'{prefix}_train.csv')
            test_dataset = load_demographic_data(participants_path,
                                                 ids_with_n_subject_pairs_dir / f'{prefix}_test.csv')

            # Initialise random seed
            np.random.seed(42)
            random.seed(42)

            train_index = train_dataset['Image_ID']
            test_index = test_dataset['Image_ID']

            x_train = kernel.loc[train_index, train_index].values
            x_test = kernel.loc[test_index, train_index].values

            y_train = train_dataset['Age'].values
            y_test = test_dataset['Age'].values

            model = SVR(kernel='precomputed')

            # Systematic search for best hyperparameters
            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
            n_nested_folds = 5
            nested_kf = KFold(n_splits=n_nested_folds, shuffle=True, random_state=i_bootstrap)
            gridsearch = GridSearchCV(model,
                                      param_grid=search_space,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_kf,
                                      verbose=0, n_jobs=1)

            gridsearch.fit(x_train, y_train)

            best_model = gridsearch.best_estimator_

            predictions = best_model.predict(x_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

            print(f'R2: {r2:0.3f}, MAE: {mae:0.3f}, RMSE: {rmse:0.3f}, CORR: {age_error_corr:0.3f}')

            # Save arrays with permutation coefs and scores as np files
            scores = np.array([r2, mae, rmse, age_error_corr])
            np.save(str(scores_dir / f'scores_{i_bootstrap:04d}_{model_name}.npy'), scores)

            train_predictions = best_model.predict(x_train)
            train_mae = mean_absolute_error(y_train, train_predictions)
            train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
            train_r2 = r2_score(y_train, train_predictions)
            train_age_error_corr, _ = stats.spearmanr(np.abs(y_train - train_predictions), y_train)

            train_scores = np.array([train_r2, train_mae, train_rmse, train_age_error_corr])
            np.save(str(scores_dir / f'scores_{i_bootstrap:04d}_{model_name}_train.npy'), train_scores)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.n_bootstrap, args.n_max_pair)
