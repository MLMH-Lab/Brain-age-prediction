#!/usr/bin/env python3
"""
Script to implement SVM and RVM in BIOBANK Scanner1 using voxel data to predict brain age.

This script assumes that a kernel has been already pre-computed. To compute the
kernel use the script `precompute_3Ddata.py`
"""
from math import sqrt
from pathlib import Path
import random
import warnings
import sys

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn_rvm import EMRVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.model_selection import GridSearchCV
import argparse

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser("Type of Vector Machine.")
parser.add_argument('vm', dest='vm_type', nargs='?', default='svm')

def main(vm_type):
    # --------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    dataset_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1'
    kernel_path = PROJECT_ROOT / 'outputs' / 'kernels' / 'kernel.csv'

    # Load demographics
    demographics = pd.read_csv((dataset_path / 'participants.tsv'), sep='\t')
    demographics.set_index('Participant_ID', inplace=True)

    # Load the Gram matrix
    kernel = pd.read_csv(kernel_path, header=0, index_col=0)

    # Compute SVM
    # --------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    if vm_type == 'svm':
        vm_dir = experiment_dir / 'voxel_SVM'
    if vm_type == 'rvm':
        vm_dir = experiment_dir / 'voxel_RVM'
    else:
        print('Only rvm and vm are acceptable as inputs for the model')
        return
    vm_dir.mkdir(exist_ok=True, parents=True)
    cv_dir = vm_dir / 'cv'
    cv_dir.mkdir(exist_ok=True, parents=True)

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
    age_predictions = pd.DataFrame(demographics['Age'])

    n_repetitions = 10
    n_folds = 10
    n_nested_folds = 5

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

            # Systematic search for best hyperparameters
            if vm_type == 'svm':
                vm = SVR(kernel='precomputed')

                search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                      2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

                nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True,
                                             random_state=i_repetition)

                gridsearch = GridSearchCV(vm,
                                          param_grid=search_space,
                                          scoring='neg_mean_absolute_error',
                                          refit=True, cv=nested_skf,
                                          verbose=3, n_jobs=1)

                gridsearch.fit(x_train, y_train)

                best_svm = gridsearch.best_estimator_

                params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                                  'params': gridsearch.cv_results_['params']}

                predictions = best_svm.predict(x_test)

            else: #rvm
                vm = EMRVR(kernel='precomputed', verbose=True)

                vm.fit(x_train, y_train)

                predictions = vm.predict(x_test)

            absolute_error = mean_absolute_error(y_test, predictions)
            root_squared_error = sqrt(mean_squared_error(y_test, predictions))
            r2_score = best_svm.score(x_test, y_test)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

            cv_r2_scores.append(r2_score)
            cv_mae.append(absolute_error)
            cv_rmse.append(root_squared_error)
            cv_age_error_corr.append(age_error_corr)

            # Save scaler, model and model parameters
            model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            params_filename = '{:02d}_{:02d}_params.joblib'.format(i_repetition, i_fold)

            dump(params_results, cv_dir / params_filename)
            dump(best_svm, cv_dir / model_filename)

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
    age_predictions.to_csv(vm_dir / 'age_predictions.csv')

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
    args = parser.parse_args(sys.argv[1:])
    main(vm_type=args.vm_type)
