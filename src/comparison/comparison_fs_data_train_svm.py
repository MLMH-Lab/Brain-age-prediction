#!/usr/bin/env python3
"""Script to train Support Vector Machines on FreeSurfer data.

We trained the Support Vector Machines (SVMs) [1] in 10 repetitions of
10 stratified k-fold cross-validation (CV) (stratified by age).
The hyperparameter tuning was performed in an automatic way using
nested CV.

References
----------
[1] - Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks."
Machine learning 20.3 (1995): 273-297.
"""
import argparse
import random
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
from joblib import dump
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    # ----------------------------------------------------------------------------------------
    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    # CV variables
    cv_r = []
    cv_r2 = []
    cv_mae = []
    cv_rmse = []
    cv_age_error_corr = []

    # Create DataFrame to hold actual and predicted ages
    age_predictions = dataset[['image_id', 'Age']]
    age_predictions = age_predictions.set_index('image_id')

    n_repetitions = 10
    n_folds = 10
    n_nested_folds = 5

    for i_repetition in range(n_repetitions):
        # Create new empty column in age_predictions df to save age predictions of this repetition
        repetition_column_name = f'Prediction repetition {i_repetition:02d}'
        age_predictions[repetition_column_name] = np.nan

        # Create 10-fold CV scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
            print(f'Running repetition {i_repetition:02d}, fold {i_fold:02d}')

            x_train, x_test = regions_norm[train_index], regions_norm[test_index]
            y_train, y_test = age[train_index], age[test_index]

            # Scaling using inter-quartile range
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            model_type = LinearSVR(loss='epsilon_insensitive')

            # Systematic search for best hyperparameters
            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
            nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)
            gridsearch = GridSearchCV(model_type,
                                      param_grid=search_space,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_skf,
                                      verbose=3, n_jobs=1)

            gridsearch.fit(x_train, y_train)

            model = gridsearch.best_estimator_

            params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                              'params': gridsearch.cv_results_['params']}

            predictions = model.predict(x_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            r, _ = stats.pearsonr(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            age_error_corr, _ = stats.spearmanr((predictions - y_test), y_test)

            cv_r.append(r)
            cv_r2.append(r2)
            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_age_error_corr.append(age_error_corr)

            # ----------------------------------------------------------------------------------------
            # Save output files
            output_prefix = f'{i_repetition:02d}_{i_fold:02d}'

            # Save scaler and model
            dump(scaler, cv_dir / f'{output_prefix}_scaler.joblib')
            dump(model, cv_dir / f'{output_prefix}_regressor.joblib')
            dump(params_results, cv_dir / f'{output_prefix}_params.joblib')

            # Save model scores
            scores_array = np.array([r, r2, mae, rmse, age_error_corr])
            np.save(cv_dir / f'{output_prefix}_scores.npy', scores_array)

            # ----------------------------------------------------------------------------------------
            # Add predictions per test_index to age_predictions
            for row, value in zip(test_index, predictions):
                age_predictions.iloc[row, age_predictions.columns.get_loc(repetition_column_name)] = value

            # Print results of the CV fold
            print(f'Repetition {i_repetition:02d} Fold {i_fold:02d} ' 
                  f'r: {r:0.3f}, R2: {r2:0.3f}, '
                  f'MAE: {mae:0.3f} RMSE: {rmse:0.3f} CORR: {age_error_corr:0.3f}')

    # Save predictions
    age_predictions.to_csv(model_dir / 'age_predictions.csv')

    # Variables for mean scores of performance metrics of CV folds across all repetitions
    print('')
    print('Mean values:')
    print(f'r: {np.mean(cv_r):0.3f} R2: {np.mean(cv_r2):0.3f} MAE: {np.mean(cv_mae):0.3f} '
          f'RMSE: {np.mean(cv_rmse):0.3f} CORR: {np.mean(cv_age_error_corr):0.3f}')


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
