#!/usr/bin/env python3
"""Permutation of SVM for BIOBANK Scanner1"""
import argparse
import random
import time
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR

from utils import COLUMNS_NAME, load_freesurfer_dataset

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

parser.add_argument('-K', '--index_min',
                    dest='index_min',
                    type=int,
                    help='index of first subject to run')

parser.add_argument('-L', '--index_max',
                    dest='index_max',
                    type=int,
                    help='index of last subject to run', )

args = parser.parse_args()


def main(experiment_name, scanner_name, input_ids_file, index_min, index_max):
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'freesurferData.csv'
    ids_path = experiment_dir / input_ids_file

    # ----------------------------------------------------------------------------------------
    permutations_dir = experiment_dir / 'permutations'
    permutations_dir.mkdir(exist_ok=True)

    dataset = load_freesurfer_dataset(participants_path, ids_path, freesurfer_path)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    n_repetitions = 10
    n_folds = 10
    n_nested_folds = 5

    # Random permutation loop
    for i_perm in range(index_min, index_max):

        # Initialise random seed
        np.random.seed(i_perm)
        random.seed(i_perm)

        # Perform permutation
        age_permuted = np.random.permutation(age)

        # Create variables to hold best model coefficients and scores per permutation
        cv_row = n_repetitions * n_folds
        n_features = regions.shape[1]
        cv_coef = np.zeros([cv_row, n_features])

        # Set i_iteration for adding coef arrays per repetition per fold to cv_coef
        i_iteration = 0

        # Create variable to hold CV scores per permutation
        cv_r2 = []
        cv_mae = []
        cv_rmse = []
        cv_age_error_corr = []

        # Loop to repeat n_folds-fold CV n_repetitions times
        for i_repetition in range(n_repetitions):
            start = time.time()

            # Create 10-fold cross-validator, stratified by age
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
            for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
                x_train, x_test = regions_norm[train_index], regions_norm[test_index]
                y_train, y_test = age_permuted[train_index], age_permuted[test_index]

                # Scaling using interquartile
                scaling = RobustScaler()
                x_train = scaling.fit_transform(x_train)
                x_test = scaling.transform(x_test)

                # Systematic search for best hyperparameters
                svm = LinearSVR(loss='epsilon_insensitive')

                search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

                nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

                gridsearch = GridSearchCV(svm,
                                          param_grid=search_space,
                                          scoring='neg_mean_absolute_error',
                                          refit=True, cv=nested_skf,
                                          verbose=1, n_jobs=1)

                gridsearch.fit(x_train, y_train)

                best_svm = gridsearch.best_estimator_

                cv_coef[i_iteration] = best_svm.coef_

                predictions = best_svm.predict(x_test)

                mae = mean_absolute_error(y_test, predictions)
                rmse = sqrt(mean_squared_error(y_test, predictions))
                r2 = best_svm.score(x_test, y_test)
                age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

                cv_r2.append(r2)
                cv_mae.append(mae)
                cv_rmse.append(rmse)
                cv_age_error_corr.append(age_error_corr)

                i_iteration += 1

                fold_time = time.time() - start
                print(f'Finished permutation {i_perm:02d} repetition {i_repetition:02d} '
                      f'fold {i_fold:02d} ETA {fold_time * (n_repetitions * n_folds - i_iteration):02f}')

        # Create np array with mean coefficients - one row per permutation, one col per feature
        cv_mean_relative_coefs = np.divide(np.abs(cv_coef), np.sum(np.abs(cv_coef), axis=1)[:, np.newaxis])
        cv_coef_mean = np.mean(cv_mean_relative_coefs, axis=0)

        # Variables for CV means across all repetitions - one row per permutation
        cv_r2_mean = np.mean(cv_r2)
        cv_mae_mean = np.mean(cv_mae)
        cv_rmse_mean = np.mean(cv_rmse)
        cv_age_error_corr_mean = np.mean(cv_age_error_corr)

        print(f'{i_perm:d}: Mean R2: {cv_r2_mean:0.3f}, MAE: {cv_mae_mean:0.3f}, RMSE: {cv_rmse_mean:0.3f}, '
              f'Error corr: {cv_age_error_corr_mean:0.3f}')

        mean_scores = np.array([cv_r2_mean, cv_mae_mean, cv_rmse_mean, cv_age_error_corr_mean])

        # Save arrays with permutation coefs and scores as np files
        filepath_coef = permutations_dir / f'perm_coef_{i_perm:04d}.npy'
        filepath_scores = permutations_dir / f'perm_scores_{i_perm:04d}.npy'
        np.save(str(filepath_coef), cv_coef_mean)
        np.save(str(filepath_scores), mean_scores)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file,
         args.index_min, args.index_max)
