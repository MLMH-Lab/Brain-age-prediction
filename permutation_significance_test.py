#!/usr/bin/env python3
"""Significance testing of SVM permutations for BIOBANK Scanner1

Ref:
Gaonkar, B., & Davatzikos, C. (2012, October). Deriving statistical significance maps for SVM based image
 classification and group comparisons. In International Conference on Medical Image Computing and Computer-Assisted
 Intervention (pp. 723-730). Springer, Berlin, Heidelberg.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.externals.joblib import load

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()
parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')
parser.add_argument('-M', '--model_name',
                    dest='model_name',
                    help='Name of the scanner.')
args = parser.parse_args()


def get_assessed_model_mean_scores(cv_dir, n_repetitions=10, n_folds=10):
    """Load scores of all SVM models run"""
    assessed_model_scores = []

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)
            assessed_model_scores.append(np.load(cv_dir / scores_filename))

    return np.asarray(assessed_model_scores)


def get_permutation_mean_scores(perm_dir, n_perm=1000):
    """Load scores of all permutations run"""
    perm_scores = []

    for i_perm in range(n_perm):

        filepath_scores = perm_dir / ('perm_scores_{:04d}.npy'.format(i_perm))

        try:
            perm_scores.append(np.load(filepath_scores))
        except FileNotFoundError:
            print('File not found: {:}'.format(filepath_scores))

    return np.asarray(perm_scores, dtype='float32')


def get_permutation_p_value(assessed_value, perm_values, greater_is_better=True):
    """Calculate p value of SVM model performance"""
    if greater_is_better:
        p_value = (np.sum(assessed_value <= perm_values) + 1.0) / (perm_values.shape[0] + 1)
    else:
        p_value = (np.sum(assessed_value >= perm_values) + 1.0) / (perm_values.shape[0] + 1)
    return p_value


def get_assessed_model_coefs(experiment_dir, n_repetitions=10, n_folds=10):
    """Load model coefficients of all SVM models run"""
    assessed_model_coefs = []

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            model = load(experiment_dir / model_filename)
            assessed_model_coefs.append(model.coef_)

    return np.asarray(assessed_model_coefs, dtype='float32')


def get_permutation_mean_relative_coefs(perm_dir, n_perm=1000):
    """Load coefficients of all permutations run"""
    perm_relative_coefs = []

    for i_perm in range(n_perm):

        filepath_scores = perm_dir / ('perm_coef_{:04d}.npy'.format(i_perm))

        try:
            perm_relative_coefs.append(np.load(filepath_scores))
        except FileNotFoundError:
            print('File not found: {:}'.format(filepath_scores))

    return np.asarray(perm_relative_coefs, dtype='float32')


def main(experiment_name, model_name):
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    cv_dir = experiment_dir / model_name / 'cv'
    perm_dir = experiment_dir / 'permutations'

    assessed_model_scores = get_assessed_model_mean_scores(cv_dir)
    perm_scores = get_permutation_mean_scores(perm_dir)

    # Assess SVM model performance by calculating p values for R2, MAE, RMSE and CORR
    score_names = ['R2', 'MAE', 'RMSE', 'CORR']
    p_value_scores = []
    for i_score, score_name in enumerate(score_names):
        if score_name in ['MAE', 'RMSE']:
            p_value = get_permutation_p_value(np.mean(assessed_model_scores[:, i_score]),
                                              perm_scores[:, i_score],
                                              greater_is_better=False)
        else:
            p_value = get_permutation_p_value(np.mean(assessed_model_scores[:, i_score]),
                                              perm_scores[:, i_score])
        p_value_scores.append(p_value)

        print('{:} : {:4.3f}'.format(score_name, p_value))

    # Save SVM model performance p values as csv
    scores_csv = pd.DataFrame([np.mean(assessed_model_scores, axis=0), p_value_scores],
                              columns=score_names,
                              index=['score', 'p value'])

    scores_csv.to_csv(perm_dir / 'scores_sig.csv')

    # Assess significance of SVM model coefficients
    assessed_model_coefs = get_assessed_model_coefs(cv_dir)
    assessed_mean_relative_coefs = np.divide(np.abs(assessed_model_coefs),
                                             np.sum(np.abs(assessed_model_coefs), axis=1)[:, np.newaxis])
    perm_mean_relative_coefs = get_permutation_mean_relative_coefs(perm_dir)

    p_value_coefs = []
    for i in range(assessed_mean_relative_coefs.shape[1]):
        p_value = get_permutation_p_value(np.mean(assessed_mean_relative_coefs[:, i]),
                                          perm_mean_relative_coefs[:, i])
        p_value_coefs.append(p_value)

    coef_csv = pd.DataFrame([np.mean(assessed_mean_relative_coefs, axis=0), p_value_coefs],
                            columns=COLUMNS_NAME,
                            index=['coefficient', 'p value'])
    coef_csv = coef_csv.transpose()
    coef_csv = coef_csv.sort_values(by=['p value'])
    coef_csv.to_csv(perm_dir / 'coefs_sig.csv')


if __name__ == "__main__":
    main(args.experiment_name, args.model_name)
