"""
Script to run SVM on bootstrap datasets of UK BIOBANK Scanner1
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    i_n_subjects = 50
    n_bootstrap = 500
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    ids_with_n_subjects_dir = experiment_dir / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subjects))

    dataset_dir = ids_with_n_subjects_dir / 'datasets'
    scores_dir = ids_with_n_subjects_dir / 'scores'
    scores_dir.mkdir(exist_ok=True)

    for i_bootstrap in range(n_bootstrap):
        print(i_bootstrap)
        dataset_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}.h5'.format(i_bootstrap, i_n_subjects)
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

        n_repetitions = 2
        n_folds = 2
        n_nested_folds = 2

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

                # Systematic search for best hyperparameters
                svm = LinearSVR(loss='epsilon_insensitive')

                # search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
                search_space = {'C': [2 ** -1]}

                nested_kf = KFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

                gridsearch = GridSearchCV(svm,
                                          param_grid=search_space,
                                          scoring='neg_mean_absolute_error',
                                          refit=True, cv=nested_kf,
                                          verbose=0, n_jobs=29)

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
        print('Mean R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}, CORR: {:0.3f}'.format(cv_r2_mean,
                                                                                    cv_mae_mean,
                                                                                    cv_rmse_mean,
                                                                                    cv_age_error_corr_mean))

        mean_scores = np.array([cv_r2_mean, cv_mae_mean, cv_rmse_mean, cv_age_error_corr_mean])

        # Save arrays with permutation coefs and scores as np files
        filepath_scores = scores_dir / ('boot_scores_{:04d}.npy'.format(i_bootstrap))
        np.save(str(filepath_scores), mean_scores)


if __name__ == "__main__":
    main()
