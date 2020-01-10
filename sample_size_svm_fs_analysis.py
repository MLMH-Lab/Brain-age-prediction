#!/usr/bin/env python3
"""Perform sample size Script to run SVM (linear SVR) on bootstrap datasets of UK BIOBANK Scanner1
IMPORTANT NOTE: This script is adapted from svm.py but uses KFold instead of StratifiedKFold
to account for the bootstrap samples with few participants
"""
from math import sqrt
from pathlib import Path
import random
import warnings

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    # ----------------------------------------------------------------------------------------

    # Loop over the 20 bootstrap samples with up to 20 gender-balanced subject pairs per age group/year
    for i_n_subject_pairs in range(1, 21):
        print('Bootstrap number of subject pairs: ', i_n_subject_pairs)
        ids_with_n_subject_pairs_dir = experiment_dir / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subject_pairs))

        dataset_dir = ids_with_n_subject_pairs_dir / 'datasets'
        scores_dir = ids_with_n_subject_pairs_dir / 'scores'
        scores_dir.mkdir(exist_ok=True)

        # Loop over the 1000 random subject samples per bootstrap
        n_bootstrap = 1000
        for i_bootstrap in range(n_bootstrap):
            print('Sample number within bootstrap: ', i_bootstrap)
            training_dataset_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}_train.h5'.format(i_bootstrap,
                                                                                                i_n_subject_pairs)
            test_dataset_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}_test.h5'.format(i_bootstrap,
                                                                                           i_n_subject_pairs)

            # Load hdf5 dataset for that bootstrap sample
            train_dataset_path = dataset_dir / training_dataset_filename
            train_dataset = pd.read_hdf(train_dataset_path, key='table')

            test_dataset_path = dataset_dir / test_dataset_filename
            test_dataset = pd.read_hdf(test_dataset_path, key='table')

            # Initialise random seed
            np.random.seed(42)
            random.seed(42)

            # Normalise regional volumes by total intracranial volume (tiv)
            regions = train_dataset[COLUMNS_NAME].values

            tiv = train_dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

            x_train = np.true_divide(regions, tiv)
            y_train = train_dataset['Age'].values

            test_tiv = test_dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]
            test_regions = test_dataset[COLUMNS_NAME].values

            x_test = np.true_divide(test_regions, test_tiv)
            y_test = test_dataset['Age'].values

            # Scaling in range [-1, 1]
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Systematic search for best hyperparameters
            n_nested_folds = 5
            svm = LinearSVR(loss='epsilon_insensitive')

            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

            nested_kf = KFold(n_splits=n_nested_folds, shuffle=True, random_state=i_bootstrap)

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

            print('Mean R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}, CORR: {:0.3f}'.format(r2_score,
                                                                                        absolute_error,
                                                                                        root_squared_error,
                                                                                        age_error_corr))

            mean_scores = np.array([r2_score, absolute_error, root_squared_error, age_error_corr])

            # Save arrays with permutation coefs and scores as np files
            filepath_scores = scores_dir / ('boot_scores_{:04d}_svm.npy'.format(i_bootstrap))
            np.save(str(filepath_scores), mean_scores)


if __name__ == "__main__":
    main()
