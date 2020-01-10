#!/usr/bin/env python3
"""Script to train Support Vector Machines on freesurfer data.

We trained the Support Vector Machines (SVMs) [1] in a 10 repetitions
10 stratified k-fold cross-validation (stratified by age).
The hyperparameter tuning was performed in an automatic way using
 a nested cross-validation.

References
----------
[1] - Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks."
Machine learning 20.3 (1995): 273-297.
"""
import argparse
from math import sqrt
from pathlib import Path
import random
import warnings

from joblib import dump
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

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
args = parser.parse_args()


def main(experiment_name, scanner_name, input_ids_file):
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'freesurferData.csv'
    ids_path = experiment_dir / input_ids_file

    model_dir = experiment_dir / 'SVM'
    model_dir.mkdir(exist_ok=True)
    cv_dir = model_dir / 'cv'
    cv_dir.mkdir(exist_ok=True)

    dataset = load_freesurfer_dataset(participants_path, ids_path, freesurfer_path)
    dataset = dataset[:1000] #TODO: remove

    # ----------------------------------------------------------------------------------------
    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    # Cross validation variables
    cv_r2_scores = []
    cv_mae = []
    cv_rmse = []
    cv_age_error_corr = []

    # Create DataFrame to hold actual and predicted ages
    age_predictions = pd.DataFrame(dataset[['Image_ID', 'Age']])
    age_predictions = age_predictions.set_index('Image_ID')

    n_repetitions = 10
    n_folds = 10
    n_nested_folds = 5

    for i_repetition in range(n_repetitions):
        # Create new empty column in age_predictions df to save age predictions of this repetition
        repetition_column_name = 'Prediction repetition {:02d}'.format(i_repetition)
        age_predictions[repetition_column_name] = np.nan

        # Create 10-fold cross-validation scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
            print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))

            x_train, x_test = regions_norm[train_index], regions_norm[test_index]
            y_train, y_test = age[train_index], age[test_index]

            # Scaling using inter-quartile
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Systematic search for best hyperparameters
            svm = LinearSVR(loss='epsilon_insensitive')

            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

            nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

            gridsearch = GridSearchCV(svm,
                                      param_grid=search_space,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_skf,
                                      verbose=3, n_jobs=1)

            gridsearch.fit(x_train, y_train)

            best_svm = gridsearch.best_estimator_

            params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                              'params': gridsearch.cv_results_['params']}

            predictions = best_svm.predict(x_test)

            absolute_error = mean_absolute_error(y_test, predictions)
            root_squared_error = sqrt(mean_squared_error(y_test, predictions))
            r2_score = best_svm.score(x_test, y_test)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

            cv_r2_scores.append(r2_score)
            cv_mae.append(absolute_error)
            cv_rmse.append(root_squared_error)
            cv_age_error_corr.append(age_error_corr)

            # Save scaler, model and model parameters
            scaler_filename = '{:02d}_{:02d}_scaler.joblib'.format(i_repetition, i_fold)
            model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            params_filename = '{:02d}_{:02d}_params.joblib'.format(i_repetition, i_fold)

            dump(scaler, cv_dir / scaler_filename)
            dump(params_results, cv_dir / params_filename)
            dump(best_svm, cv_dir / model_filename)

            # Save model scores r2, MAE, RMSE, correlation between error and age
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
