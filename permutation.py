"""Permutation of SVM for BIOBANK Scanner1"""
from math import sqrt
from pathlib import Path
import random
import time
import warnings

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')


def main(args):
    # Create output subdirectory
    experiment_name = 'total'
    perm_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'permutations'
    perm_dir.mkdir(exist_ok=True)

    # Load hdf5 file
    file_name = 'freesurferData_' + experiment_name + '.h5'
    dataset = pd.read_hdf(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / file_name, key='table')

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    n_features = regions.shape[1]

    n_repetitions = 1
    n_folds = 3
    n_nested_folds = 3

    # Random permutation loop
    for i_perm in range(args.index_min, args.index_max):

        # Initialise random seed
        np.random.seed(i_perm)
        random.seed(i_perm)

        # Perform permutation
        age_permuted = np.random.permutation(age)

        # Create variables to hold best model coefficients and scores per permutation
        cv_row = n_repetitions * n_folds
        cv_coef = np.zeros([cv_row, n_features])

        # Set i_iteration for adding coef arrays per repetition per fold to cv_coef
        i_iteration = 0

        # Create variable to hold CV scores per permutation
        cv_r2_scores = np.array([[]])
        cv_mae = np.array([[]])
        cv_rmse = np.array([[]])

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

                absolute_error = mean_absolute_error(y_test, predictions)
                root_squared_error = sqrt(mean_squared_error(y_test, predictions))
                r2_score = best_svm.score(x_test, y_test)

                cv_r2_scores = np.append(cv_r2_scores, r2_score)
                cv_mae = np.append(cv_mae, absolute_error)
                cv_rmse = np.append(cv_rmse, root_squared_error)

                i_iteration += 1

                fold_time = time.time() - start
                print('Finished permutation {:02d}, repetition :{02d}, fold {:02d}, ETA {f}'
                      .format(i_perm, i_repetition, i_fold, fold_time * (n_repetitions * n_folds - i_iteration)))

        # Create np array with mean coefficients - one row per permutation, one col per feature
        cv_coef_abs = np.abs(cv_coef)
        cv_coef_mean = np.mean(cv_coef_abs, axis=0)

        # Variables for CV means across all repetitions - one row per permutation
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_mae_mean = np.mean(cv_mae)
        cv_rmse_mean = np.mean(cv_rmse)

        print('{:d}: Mean R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}'
              .format(i_perm, cv_r2_mean, cv_mae_mean, cv_rmse_mean))

        mean_scores = np.array([cv_r2_mean, cv_mae_mean, cv_rmse_mean])

        # Save arrays with permutation coefs and scores as np files
        filepath_coef = perm_dir / ('perm_coef_{:04d}.npy'.format(i_perm))
        filepath_scores = perm_dir / ('perm_scores_{:04d}.npy'.format(i_perm))
        np.save(str(filepath_coef), cv_coef_mean)
        np.save(str(filepath_scores), mean_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('index_min', help='index of first subject to run', type=int)
    parser.add_argument('index_max', help='index of last subject to run', type=int)
    args = parser.parse_args()

    main(args)
