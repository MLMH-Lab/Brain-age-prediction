#!/usr/bin/env python3
"""Comparing classifiers using a version of the paired Student’s t-test that is
corrected for the violation of the independence assumption from repeated k-fold cross-validation
when training the model

Based on:
https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/

Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004.
https://github.com/BayesianTestsML/tutorial/blob/9fb0bf75b4435d61d42935be4d0bfafcc43e77b9/Python/bayesiantests.py
"""
import argparse
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path.cwd()


def ttest_ind_corrected(performance_a, performance_b, k=10, r=10):
    """Corrected repeated k-fold cv test.
     The test assumes that the classifiers were evaluated using cross validation.

    Ref:
        Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning
         algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004

    Args:
        performance_a: performances from classifier A
        performance_b: performances from classifier B
        k: number of folds
        r: number of repetitions

    Returns:
         t: t-statistic of the corrected test.
         prob: p-value of the corrected test.
    """
    df = k * r - 1

    x = performance_a - performance_b
    m = np.mean(x)

    sigma_2 = np.var(x, ddof=1)
    denom = np.sqrt((1 / k * r + 1 / (k - 1)) * sigma_2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(m, denom)

    prob = stats.t.sf(np.abs(t), df) * 2

    return t, prob


def main(experiment_name, suffix, model_list):
    # Create summary of results
    n_repetitions = 10
    n_folds = 10
    for model_name in model_list:
        model_dir = PROJECT_ROOT / 'outputs' / experiment_name / model_name
        cv_dir = model_dir / 'cv'

        r2_scores = []
        absolute_errors = []
        root_squared_errors = []
        age_error_corrs = []

        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)
                r2_score, absolute_error, root_squared_error, age_error_corr = np.load(cv_dir / scores_filename)
                r2_scores.append(r2_score)
                absolute_errors.append(absolute_error)
                root_squared_errors.append(root_squared_error)
                age_error_corrs.append(age_error_corr)

        results = pd.DataFrame(columns=['Measure', 'Value'])
        results = results.append({'Measure': 'mean_r2', 'Value': np.mean(r2_scores)}, ignore_index=True)
        results = results.append({'Measure': 'std_r2', 'Value': np.std(r2_scores)}, ignore_index=True)
        results = results.append({'Measure': 'mean_mae', 'Value': np.mean(absolute_errors)}, ignore_index=True)
        results = results.append({'Measure': 'std_mae', 'Value': np.std(absolute_errors)}, ignore_index=True)
        results = results.append({'Measure': 'mean_rmse', 'Value': np.mean(root_squared_errors)}, ignore_index=True)
        results = results.append({'Measure': 'std_rmse', 'Value': np.std(root_squared_errors)}, ignore_index=True)
        results = results.append({'Measure': 'mean_age_error_corr', 'Value': np.mean(age_error_corrs)}, ignore_index=True)
        results = results.append({'Measure': 'std_age_error_corr', 'Value': np.std(age_error_corrs)}, ignore_index=True)
        results.to_csv(model_dir / '{:}_scores_summary.csv'.format(model_name), index=False)

    combinations = list(itertools.combinations(model_list, 2))

    # Bonferroni correction for multiple comparisons
    corrected_alpha = 0.05 / len(combinations)

    results_df = pd.DataFrame(columns=['regressors', 'p-value', 'stats'])

    for classifier_a, classifier_b in combinations:
        classifier_a_dir = PROJECT_ROOT / 'outputs' / experiment_name / classifier_a
        classifier_b_dir = PROJECT_ROOT / 'outputs' / experiment_name / classifier_b

        mae_a = []
        mae_b = []

        n_repetitions = 10
        n_folds = 10

        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)

                performance_a = np.load(classifier_a_dir / 'cv' / scores_filename)[1]
                performance_b = np.load(classifier_b_dir / 'cv' / scores_filename)[1]

                mae_a.append(performance_a)
                mae_b.append(performance_b)

        statistic, pvalue = ttest_ind_corrected(np.asarray(mae_a), np.asarray(mae_b), k=n_folds, r=n_repetitions)

        print('{} vs. {} pvalue: {:6.3}'.format(classifier_a, classifier_b, pvalue), end='')
        if pvalue <= corrected_alpha:
            print('*')
        else:
            print('')

        results_df = results_df.append({'regressors': '{} vs. {}'.format(classifier_a, classifier_b),
                                        'p-value': pvalue,
                                        'stats': statistic},
                                       ignore_index=True)

        results_df.to_csv(PROJECT_ROOT / 'outputs' / experiment_name / ('regressors_comparison' + suffix + '.csv'),
                          index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-E', '--experiment_name',
                        dest='experiment_name',
                        help='Experiment name where the model predictions are stored.')

    parser.add_argument('-S', '--suffix',
                        dest='suffix',
                        help='Suffix to add on the output file regressors_comparison_suffix.csv.')

    parser.add_argument('-M', '--model_list',
                        dest='model_list',
                        nargs='+',
                        help='Names of models to analyse.')

    args = parser.parse_args()

    main(args.experiment_name, args.suffix, args.model_list)
