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
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main(args):
    # Define what subjects dataset should contain: total, male or female
    subjects = 'total'

    # Load hdf5 file, use rows specified in arguments only
    file_name = 'freesurferData_' + subjects + '.h5'
    dataset = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1' / file_name, key='table')

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[dataset.columns[5:-2]].values
    tiv = dataset.EstimatedTotalIntraCranialVol.values
    tiv = tiv.reshape(len(dataset), 1)
    regions_norm = np.true_divide(regions, tiv)  # Independent vars X
    age = dataset[dataset.columns[2]].values  # Dependent var Y

    n_repetitions = 2
    n_folds = 2
    n_nested_folds = 2

    # initialise np arrays for saving coefficients and scores (one row per i_perm)
    array_coef = np.zeros([1,100])
    # TODO: what is a better way to create an empty array with the right shape without havign to delete the first row?
    array_coef = np.delete(array_coef, 0, 0)
    array_scores = np.array([[]])

    # Random permutation loop
    # for i_perm in range(args.index_min, args.index_max):
    for i_perm in range(2):

        np.random.seed(i_perm)
        random.seed(i_perm)

        age_permuted = np.random.permutation(age)

        # Create variable to hold best model coefficients per permutation
        cv_coef = np.zeros([1,100])
        cv_coef = np.delete(cv_coef, 0, 0)

        # Create variable to hold CV scores per permutation
        cv_r2_scores = np.array([[]])
        cv_mae = np.array([[]])
        cv_rmse = np.array([[]])

        # Loop to repeat 10-fold CV 10 times
        for i_repetition in range(n_repetitions):

            # Create 10-fold cross-validator, stratified by age
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)

            for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
                print('Running repetition %02d, fold %02d' % (i_repetition, i_fold))

                x_train, x_test = regions_norm[train_index], regions_norm[test_index]
                y_train, y_test = age_permuted[train_index], age_permuted[test_index]

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
                cv_coef = np.concatenate((cv_coef, coef), axis=0)

                predictions = gridsearch.predict(x_test)

                absolute_error = mean_absolute_error(y_test, predictions)
                root_squared_error = sqrt(mean_squared_error(y_test, predictions))
                r2_score = svm_train_best.score(x_test, y_test)

                cv_r2_scores = np.append(cv_r2_scores, r2_score)
                cv_mae = np.append(cv_mae, absolute_error)
                cv_rmse = np.append(cv_rmse, root_squared_error)

        # Create np array with mean coefficients - one row per permutation, one col per FS region
        cv_coef_abs = np.abs(cv_coef)
        cv_coef_mean = np.mean(cv_coef_abs, axis=0)
        cv_coef_mean = cv_coef_mean[np.newaxis, :]
        array_coef = np.concatenate((array_coef, cv_coef_mean), axis=0)

        # Variables for CV means across all repetitions - save one mean per permutation
        # should the below use absolute values?
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_mae_mean = np.mean(cv_mae)
        cv_rmse_mean = np.mean(cv_rmse)
        print('Mean R2: %0.3f, MAE: %0.3f, RMSE: %0.3f' % (cv_r2_mean, cv_mae_mean, cv_rmse_mean))
        # the 3 lines below are not working right
        mean_scores = np.array([cv_r2_mean, cv_mae_mean, cv_rmse_mean])
        mean_scores = mean_scores[np.newaxis, :]
        array_scores = np.concatenate((array_scores, mean_scores), axis=1)

    # Save arrays with permutation coefs and scores as np files - NOT CHECKED YET
    filepath_coef = '/home/lea/PycharmProjects/predicted_brain_age/outputs/permutations/total/perm_coef.npy'
    filepath_scores = '/home/lea/PycharmProjects/predicted_brain_age/outputs/permutations/total/perm_scores.npy'
    np.save(filepath_coef, array_coef)
    np.save(filepath_scores, array_scores)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("index_min", help="index of first subject to run",
                        type=int)
    parser.add_argument("index_max", help="index of last subject to run",
                        type=int)
    args = parser.parse_args()

    main(args)
