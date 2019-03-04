"""Permutation of SVM for BIOBANK Scanner1"""

from math import sqrt
from pathlib import Path
import random

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main(args):


    # Define what subjects dataset should contain: total, male or female
    subjects = 'total'

    # Load hdf5 file, use rows specified in arguments only
    file_name = 'freesurferData_' + subjects + '.h5'
    dataset = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1' / file_name, key='table')
    dataset = dataset[args.index_min:args.index_max]

    # Initialise random seed
    np.random.seed = 42
    random.seed = 42

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[dataset.columns[5:-2]].values
    tiv = dataset.EstimatedTotalIntraCranialVol.values
    tiv = tiv.reshape(len(dataset), 1)
    regions_norm = np.true_divide(regions, tiv)  # Independent vars X
    age = dataset[dataset.columns[2]].values  # Dependent var Y

    n_repetitions = 3
    n_folds = 3
    n_nested_folds = 3
    n_perm = 3 # increase to 10000 once the script is done

    # Random permutation loop
    for i_perm in range(n_perm):

        np.random.seed = i_perm
        random.seed = i_perm
        perm = np.random.permutation(age.size)
        age_perm = age[perm]

        # intitialise np arrays for saving coefficients and scores (one row per i_perm)
        array_coef = np.array([])
        array_scores = np.array([])

        # Create variable to hold best model coefficients per permutation
        cv_coef = []

        # Create variable to hold CV scores per permutation
        cv_r2_scores = []
        cv_mae = []
        cv_rmse = []

        # Loop to repeat 10-fold CV 10 times
        for i_repetition in range(n_repetitions):

            # Create 10-fold cross-validator, stratified by age
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)

            for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
                print('Running repetition %02d, fold %02d' % (i_repetition, i_fold))

                x_train, x_test = regions_norm[train_index], regions_norm[test_index]
                y_train, y_test = age_perm[train_index], age_perm[test_index]

                # Scaling in range [-1, 1]
                scaling = MinMaxScaler(feature_range=(-1, 1))
                x_train = scaling.fit_transform(x_train)
                x_test = scaling.transform(x_test)

                # Systematic search for best hyperparameters
                svm = SVR(kernel='linear')

                c_range = [0.001, 0.01, 0.1] # shortened for testing
                # c_range = [0.001, 0.01, 0.1, 1, 10, 100]
                search_space = [{'C': c_range}]
                nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

                gridsearch = GridSearchCV(svm, param_grid=search_space, refit=True, cv=nested_skf, verbose=3)
                gridsearch.fit(x_train, y_train)
                svm_train_best = gridsearch.best_estimator_
                coef = svm_train_best.coef_
                cv_coef = cv_coef.append(coef)

                predictions = gridsearch.predict(x_test)
                absolute_error = mean_absolute_error(y_test, predictions)
                root_squared_error = sqrt(mean_squared_error(y_test, predictions))
                r2_score = svm_train_best.score(x_test, y_test)
                cv_r2_scores.append(r2_score)
                cv_mae.append(absolute_error)
                cv_rmse.append(root_squared_error)

        # Create np array with mean coefficients - one row per permutation, one col per FS region
        cv_coef_mean = np.mean(cv_coef)
        if array_coef.size == 0:
            array_coef = cv_coef_mean
        else:
            np.vstack(array_coef, cv_coef_mean)

        # Variables for CV means across all repetitions - save one mean per permutation
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_mae_mean = np.mean(cv_mae)
        cv_rmse_mean = np.mean(cv_rmse)
        print('Mean R2: %0.3f, MAE: %0.3f, RMSE: %0.3f' % (cv_r2_mean, cv_mae_mean, cv_rmse_mean))
        mean_scores = np.concatenate((cv_r2_mean, cv_mae_mean, cv_rmse_mean))
        if array_scores.size == 0:
            array_scores = mean_scores
        else:
            np.vstack(array_scores, mean_scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("index_min", help="index of first subject to run",
                        type=int)
    parser.add_argument("index_max", help="index of last subject to run",
                        type=int)
    args = parser.parse_args()

    main(args)
