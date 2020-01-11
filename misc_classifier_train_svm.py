#!/usr/bin/env python3
"""
Script to run SVM classifier (linear SVC) on bootstrap dataset (with i_n_subject_pairs = 50) of UK BIOBANK Scanner1.
The obtained scores will be compared with the scores from the regressor (with i_n_subject_pairs = 50).
"""
import random
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    # ----------------------------------------------------------------------------------------
    i_n_subject_pairs = 50

    print('Bootstrap number of subject pairs: ', i_n_subject_pairs)

    ids_with_n_subjects_dir = experiment_dir / 'bootstrap_analysis' / f'{i_n_subject_pairs:02d}'

    dataset_dir = ids_with_n_subjects_dir / 'datasets'
    scores_dir = ids_with_n_subjects_dir / 'scores_classifier'
    scores_dir.mkdir(exist_ok=True)

    # Loop over the 1000 random subject samples per bootstrap
    n_bootstrap = 1000
    for i_bootstrap in range(n_bootstrap):
        print(f'Sample number within bootstrap: {i_bootstrap}')
        dataset_filename = f'homogeneous_bootstrap_{i_bootstrap:04d}_n_{i_n_subject_pairs:02d}.h5'
        dataset_path = dataset_dir / dataset_filename

        # Load hdf5 dataset for that bootstrap sample
        dataset = pd.read_hdf(dataset_path, key='table')

        # Initialise random seed
        np.random.seed(42)
        random.seed(42)

        # Normalise regional volumes by total intracranial volume (tiv)
        regions = dataset[COLUMNS_NAME].values

        tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

        regions_norm = np.true_divide(regions, tiv)
        age = dataset['Age'].values

        # Create variable to hold CV variables
        cv_r2_scores = []
        cv_mae = []
        cv_rmse = []
        cv_age_error_corr = []

        n_repetitions = 10
        n_folds = 10
        n_nested_folds = 5

        for i_repetition in range(n_repetitions):
            # Create 10-fold cross-validator
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
            for i_fold, (train_index, test_index) in enumerate(kf.split(regions_norm, age)):
                x_train, x_test = regions_norm[train_index], regions_norm[test_index]
                y_train, y_test = age[train_index], age[test_index]

                # Scaling in range [-1, 1]
                scaler = RobustScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                svm = LinearSVC(loss='hinge')

                # Systematic search for best hyperparameters
                search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
                nested_kf = KFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)
                gridsearch = GridSearchCV(svm,
                                          param_grid=search_space,
                                          scoring='neg_mean_absolute_error',
                                          refit=True, cv=nested_kf,
                                          verbose=0, n_jobs=1)

                gridsearch.fit(x_train, y_train)

                best_svm = gridsearch.best_estimator_

                predictions = best_svm.predict(x_test)

                absolute_error = mean_absolute_error(y_test, predictions)
                root_squared_error = sqrt(mean_squared_error(y_test, predictions))
                r2_score = best_svm.score(x_test, y_test)
                age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

                cv_r2_scores.append(r2_score)
                cv_mae.append(absolute_error)
                cv_rmse.append(root_squared_error)
                cv_age_error_corr.append(age_error_corr)

        # Variables for CV means across all repetitions
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_mae_mean = np.mean(cv_mae)
        cv_rmse_mean = np.mean(cv_rmse)
        cv_age_error_corr_mean = np.mean(np.abs(cv_age_error_corr))
        print(f'Mean R2: {cv_r2_mean:0.3f}, MAE: {cv_mae_mean:0.3f}, RMSE: {cv_rmse_mean:0.3f}, '
              f'CORR: {cv_age_error_corr_mean:0.3f}')

        mean_scores = np.array([cv_r2_mean, cv_mae_mean, cv_rmse_mean, cv_age_error_corr_mean])

        # Save arrays with permutation coefs and scores as np files
        filepath_scores = scores_dir / f'boot_scores_{i_bootstrap:04d}.npy'
        np.save(str(filepath_scores), mean_scores)


if __name__ == '__main__':
    main()
