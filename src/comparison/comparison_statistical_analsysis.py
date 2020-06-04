#!/usr/bin/env python3
"""Comparing classifiers using a version of the paired Student’s t-test that is
corrected for the violation of the independence assumption from repeated k-fold cross-validation
when training the model

References:
-----------
[1] - https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/

[2] - Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing
 learning algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin,
 Heidelberg, 2004.

[3] - https://github.com/BayesianTestsML/tutorial/blob/9fb0bf75b4435d61d42935be4d0bfafcc43e77b9/Python/bayesiantests.py
"""
import argparse
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from utils import ttest_ind_corrected

PROJECT_ROOT = Path.cwd()

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


def main(experiment_name, suffix, model_list):
    # Create summary of results
    n_repetitions = 10
    n_folds = 10
    for model_name in model_list:
        model_dir = PROJECT_ROOT / 'outputs' / experiment_name / model_name
        cv_dir = model_dir / 'cv'

        r2_list = []
        mae_list = []
        rmse_list = []
        age_error_corr_list = []

        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                r2, mae, rmse, age_error_corr = np.load(cv_dir / f'{i_repetition:02d}_{i_fold:02d}_scores.npy')
                r2_list.append(r2)
                mae_list.append(mae)
                rmse_list.append(rmse)
                age_error_corr_list.append(age_error_corr)

        results = pd.DataFrame(columns=['Measure', 'Value'])
        results = results.append({'Measure': 'mean_r2', 'Value': np.mean(r2_list)}, ignore_index=True)
        results = results.append({'Measure': 'std_r2', 'Value': np.std(r2_list)}, ignore_index=True)
        results = results.append({'Measure': 'mean_mae', 'Value': np.mean(mae_list)}, ignore_index=True)
        results = results.append({'Measure': 'std_mae', 'Value': np.std(mae_list)}, ignore_index=True)
        results = results.append({'Measure': 'mean_rmse', 'Value': np.mean(rmse_list)}, ignore_index=True)
        results = results.append({'Measure': 'std_rmse', 'Value': np.std(rmse_list)}, ignore_index=True)
        results = results.append({'Measure': 'mean_age_error_corr', 'Value': np.mean(age_error_corr_list)},
                                 ignore_index=True)
        results = results.append({'Measure': 'std_age_error_corr', 'Value': np.std(age_error_corr_list)},
                                 ignore_index=True)

        results.to_csv(model_dir / f'{model_name}_scores_summary.csv', index=False)

    combinations = list(itertools.combinations(model_list, 2))

    # Bonferroni correction for multiple comparisons
    corrected_alpha = 0.05 / len(combinations)

    results_df = pd.DataFrame(columns=['regressors', 'p-value', 'stats'])

    for classifier_a, classifier_b in combinations:
        classifier_a_dir = PROJECT_ROOT / 'outputs' / experiment_name / classifier_a
        classifier_b_dir = PROJECT_ROOT / 'outputs' / experiment_name / classifier_b

        mae_a = []
        mae_b = []

        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                performance_a = np.load(classifier_a_dir / 'cv' / f'{i_repetition:02d}_{i_fold:02d}_scores.npy')[1]
                performance_b = np.load(classifier_b_dir / 'cv' / f'{i_repetition:02d}_{i_fold:02d}_scores.npy')[1]

                mae_a.append(performance_a)
                mae_b.append(performance_b)

        statistic, pvalue = ttest_ind_corrected(np.asarray(mae_a), np.asarray(mae_b), k=n_folds, r=n_repetitions)

        print(f'{classifier_a} vs. {classifier_b} pvalue: {pvalue:6.3}', end='')
        if pvalue <= corrected_alpha:
            print('*')
        else:
            print('')

        results_df = results_df.append({'regressors': f'{classifier_a} vs. {classifier_b}',
                                        'p-value': pvalue,
                                        'stats': statistic},
                                       ignore_index=True)

        results_df.to_csv(PROJECT_ROOT / 'outputs' / experiment_name / f'regressors_comparison{suffix}.csv',
                          index=False)


if __name__ == '__main__':
    main(args.experiment_name, args.suffix, args.model_list)
