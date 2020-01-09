"""Permutation of SVM for BIOBANK Scanner1"""
from math import sqrt
from pathlib import Path
import random
import time
import warnings

from scipy import stats
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn_rvm.em_rvm import EMRVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')


def main(index_min, index_max):
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    dataset_path = PROJECT_ROOT / 'outputs' / experiment_name / 'freesurferData.h5'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    permutations_dir = experiment_dir / 'permutations'
    permutations_dir.mkdir(exist_ok=True)

    # Load hdf5 file
    dataset = pd.read_hdf(dataset_path, key='table')

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    n_repetitions = 10
    n_folds = 10

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
        cv_r2_scores = []
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
                rvm = EMRVR(kernel='linear')
                rvm.fit(x_train, y_train)
                predictions = rvm.predict(x_test)

                absolute_error = mean_absolute_error(y_test, predictions)
                root_squared_error = sqrt(mean_squared_error(y_test, predictions))
                r2_score = rvm.score(x_test, y_test)
                age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

                cv_r2_scores.append(r2_score)
                cv_mae.append(absolute_error)
                cv_rmse.append(root_squared_error)
                cv_age_error_corr.append(age_error_corr)

                i_iteration += 1

                fold_time = time.time() - start
                print('Finished permutation {:02d}, repetition {:02d}, fold {:02d}, ETA {:02f}'
                      .format(i_perm, i_repetition, i_fold, fold_time * (n_repetitions * n_folds - i_iteration)))

        # Create np array with mean coefficients - one row per permutation, one col per feature
        cv_mean_relative_coefs = np.divide(np.abs(cv_coef), np.sum(np.abs(cv_coef), axis=1)[:, np.newaxis])
        cv_coef_mean = np.mean(cv_mean_relative_coefs, axis=0)

        # Variables for CV means across all repetitions - one row per permutation
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_mae_mean = np.mean(cv_mae)
        cv_rmse_mean = np.mean(cv_rmse)
        cv_age_error_corr_mean = np.mean(cv_age_error_corr)

        print('{:d}: Mean R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}, Error corr: {:0.3f}'
              .format(i_perm, cv_r2_mean, cv_mae_mean, cv_rmse_mean, cv_age_error_corr_mean))

        mean_scores = np.array([cv_r2_mean, cv_mae_mean, cv_rmse_mean, cv_age_error_corr_mean])

        # Save arrays with permutation coefs and scores as np files
        filepath_coef = permutations_dir / ('perm_coef_{:04d}.npy'.format(i_perm))
        filepath_scores = permutations_dir / ('perm_scores_{:04d}.npy'.format(i_perm))
        np.save(str(filepath_coef), cv_coef_mean)
        np.save(str(filepath_scores), mean_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('index_min', help='index of first subject to run', type=int)
    parser.add_argument('index_max', help='index of last subject to run', type=int)
    args = parser.parse_args()

    main(args.index_min, args.index_max)