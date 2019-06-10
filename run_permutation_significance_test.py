"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np
import pandas as pd
import os

from sklearn.externals import joblib

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()


def get_assessed_model_mean_scores(cv_dir, n_repetitions=10, n_folds=10):
    """"""
    assessed_model_scores = []

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            scores_filename = '{:02d}_{:02d}_svm_scores.joblib'.format(i_repetition, i_fold)
            assessed_model_scores.append(np.load(cv_dir / scores_filename))

    return np.asarray(assessed_model_scores, dtype='float32')


def get_permutation_mean_scores(perm_dir, n_perm=1000):
    """"""
    perm_scores = []

    for i_perm in range(n_perm):

        filepath_scores = perm_dir / ('perm_scores_{:04d}.npy'.format(i_perm))

        try:
            perm_scores.append(np.load(filepath_scores))
        except FileNotFoundError:
            print('File not found: {:}'.format(filepath_scores))

    return np.asarray(perm_scores, dtype='float32')


def get_permutation_p_value(assessed_score, perm_scores):
    """"""
    return (np.sum(assessed_score >= perm_scores) + 1.0) / (perm_scores.shape[0] + 1)


def get_assessed_model_coefs(experiment_dir, n_repetitions = 10, n_folds = 10):
    """"""
    model_coefs = []

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            model_file_name = '{:02d}_{:02d}_svm.joblib'.format(i_repetition, i_fold)
            model = np.load(experiment_dir / model_file_name)
            model_coefs.append(model.coef_)

    return np.asarray(model_coefs, dtype='float32')


def get_permutation_mean_relative_coefs(perm_dir, n_perm=1000):
    """"""
    perm_relative_coefs = []

    for i_perm in range(n_perm):

        filepath_scores = perm_dir / ('perm_coef_{:04d}.npy'.format(i_perm))

        try:
            perm_relative_coefs.append(np.load(filepath_scores))
        except FileNotFoundError:
            print('File not found: {:}'.format(filepath_scores))

    return np.asarray(perm_relative_coefs, dtype='float32')


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / 'SVM'
    cv_dir = svm_dir / 'cv'
    perm_dir = experiment_dir / 'permutations'

    model_scores = get_assessed_model_mean_scores(cv_dir)
    perm_scores = get_permutation_mean_scores(perm_dir)

    # Perform
    scores_names = ['R2', 'MAE', 'RMSE']
    p_value_scores = []
    for (score_name, i_score) in zip(scores_names, range(model_scores.shape[1])):
        p_value = get_permutation_p_value(np.mean(model_scores[:, i_score]), perm_scores[:,i_score])
        p_value_scores.append(p_value)

        print('{:} : {:4.3f}'.format(score_name, p_value))

    scores_csv = pd.DataFrame([np.mean(model_scores, axis=0), p_value_scores],
                              columns=scores_names,
                              index=['score', 'p value'])
    scores_csv.to_csv(PROJECT_ROOT / 'outputs' / experiment_name / 'scores_sig.csv')

    # Perform x
    model_coef = get_assessed_model_coefs(experiment_dir)
    model_coef = np.abs(model_coef) / np.sum(np.abs(model_coef), axis=-1)
    perm_relative_coefs = get_permutation_mean_relative_coefs(perm_dir)

    p_value_coef = []
    for i in range(model_coef.shape[0]):
        p_value = get_permutation_p_value(np.mean(model_coef[:, i_score]), perm_relative_coefs[:, i_score])
        p_value_coef.append(p_value)

    # Save as csv
    coef_csv = pd.DataFrame([np.mean(model_coef, axis=0), p_value_coef],
                            columns=COLUMNS_NAME,
                            index=['coefficient', 'p value'])
    coef_csv = coef_csv.transpose()
    coef_csv = coef_csv.sort_values(by=['p value'])
    coef_csv.to_csv(PROJECT_ROOT / 'outputs' / experiment_name / 'coef_sig.csv')


if __name__ == "__main__":
    main()