#!/usr/bin/env python3
"""
Compares performance of SVM models with different hyperparameters C.
"""

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from regressors_comparison import ttest_ind_corrected

PROJECT_ROOT = Path.cwd()


def main():
    """Pairwise comparison of SVM classifier performances with different hyperparameters."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    # ----------------------------------------------------------------------------------------
    svm_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'SVM'
    cv_dir = svm_dir / 'cv'

    n_repetitions = 10
    n_folds = 10

    search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

    scores_params = []

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            params_filename = f'{i_repetition:02d}_{i_fold:02d}_params.joblib'
            params_dict = joblib.load(cv_dir / params_filename)

            scores_params.append(params_dict['means'])

    scores_params = np.array(scores_params)

    combinations = list(itertools.combinations(range(scores_params.shape[1]), 2))

    # Bonferroni correction for multiple comparisons
    corrected_alpha = 0.05 / len(combinations)

    results_df = pd.DataFrame(columns=['params', 'p-value', 'stats'])

    # Corrected repeated k-fold cv test to compare performance of two SVM classifiers at a time
    for param_a, param_b in combinations:
        statistic, pvalue = ttest_ind_corrected(scores_params[:, param_a], scores_params[:, param_b],
                                                k=n_folds,
                                                r=n_repetitions)

        print('{} vs. {} pvalue: {:6.3}'.format(search_space['C'][param_a], search_space['C'][param_b], pvalue), end='')
        if pvalue <= corrected_alpha:
            print('*')
        else:
            print('')

        results_df = results_df.append({'params': '{} vs. {}'.format(search_space['C'][param_a],
                                                                     search_space['C'][param_b]),
                                        'p-value': pvalue,
                                        'stats': statistic},
                                       ignore_index=True)

    results_df.to_csv(svm_dir / 'svm_params_ttest.csv', index=False)

    values_df = pd.DataFrame(columns=['measures'] + list(search_space['C']))

    scores_params_mean = np.mean(scores_params, axis=0)
    scores_params_std = np.std(scores_params, axis=0)

    values_df.loc[0] = ['mean'] + list(scores_params_mean)
    values_df.loc[1] = ['std'] + list(scores_params_std)

    values_df.to_csv(svm_dir / 'svm_params_values.csv', index=False)


if __name__ == '__main__':
    main()
