#!/usr/bin/env python3
"""Script to assess correlations between BrainAGE/BrainAGER and demographic variables in UK Biobank
(dataset created in variables_biobank.py)"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_ind

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-M', '--model_name',
                    dest='model_name',
                    help='Name of the model.')

args = parser.parse_args()

UNIVERSITY_CODE = 4
PROFESSIONAL_QUAL_CODE = 3
A_LEVEL_CODE = 2
GCSE_CODE = 1


def spearman(df, x, y):
    """Calculate and interpret Spearman's rank-order correlation of variables x and y"""

    spearman_rho, spearman_p = spearmanr(df[x], df[y])

    alpha = 0.05
    n = len(df)

    if spearman_p < alpha:
        print(f'n={n}, {x} and {y} - reject H0: p = {spearman_p:6.3}, rho = {spearman_rho:6.3}')
    else:
        print(f'n={n}, {x} and {y} - not significant: p = {spearman_p:6.3}, rho = {spearman_rho:6.3}')


def cohend(d1, d2):
    """Calculate Cohen's d effect size for independent samples"""

    n1, n2 = len(d1), len(d2)
    d1_variance, d2_variance = np.var(d1, ddof=1), np.var(d2, ddof=1)
    std_pooled = np.sqrt(((n1 - 1) * d1_variance + (n2 - 1) * d2_variance) / (n1 + n2 - 2))
    d1_mean, d2_mean = np.mean(d1), np.mean(d2)
    effect_size = (d1_mean - d2_mean) / std_pooled

    return effect_size


def main(experiment_name, model_name):
    """"""
    correlation_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'correlation_analysis'

    ensemble_df = pd.read_csv(correlation_dir / f'ensemble_{model_name}_output.csv')
    ensemble_df['id'] = ensemble_df['Image_ID'].str.split('_').str[0]
    ensemble_df['id'] = ensemble_df['id'].str.split('-').str[1]
    ensemble_df['id'] = pd.to_numeric(ensemble_df['id'])

    covariates_df = pd.read_csv(PROJECT_ROOT / 'outputs' / 'covariates' / 'covariates.csv')

    dataset = pd.merge(ensemble_df, covariates_df, on='id')

    # Correlation variables
    y_list = ['BrainAGE_predmean', 'BrainAGER_predmean']
    x_list = ['Air_pollution',
              'Traffic_intensity', 'Inverse_dist_road',
              'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Create empty dataframe to store correlation results
    corr_output = pd.DataFrame({'Row_labels_1': ['Air_pollution', 'Air_pollution', 'Air_pollution',
                                                 'Traffic_intensity', 'Traffic_intensity', 'Traffic_intensity',
                                                 'Inverse_dist_road', 'Inverse_dist_road', 'Inverse_dist_road',
                                                 'Greenspace_perc', 'Greenspace_perc', 'Greenspace_perc',
                                                 'Garden_perc', 'Garden_perc', 'Garden_perc',
                                                 'Water_perc', 'Water_perc', 'Water_perc',
                                                 'Natural_env_perc', 'Natural_env_perc', 'Natural_env_perc'],
                                'Row_labels_2': ['n', 'p_val', 'rho',
                                                 'n', 'p_val', 'rho',
                                                 'n', 'p_val', 'rho',
                                                 'n', 'p_val', 'rho',
                                                 'n', 'p_val', 'rho',
                                                 'n', 'p_val', 'rho',
                                                 'n', 'p_val', 'rho']})
    corr_output.set_index('Row_labels_1', 'Row_labels_2')

    # Spearman's correlation per variable, add to corr_output
    for y in y_list:
        y_values = []
        for x in x_list:
            dataset_x = dataset.dropna(subset=[x])
            n = len(dataset_x)
            spearman_rho, spearman_p = spearmanr(dataset_x[x], dataset_x[y])
            y_values.append(n)
            y_values.append(spearman_p)
            y_values.append(spearman_rho)
        corr_output[y] = y_values

    corr_output.to_csv(correlation_dir / f'covariates_spearman_{model_name}output.csv')

    # Create empty dataframe for analysis of education level
    education_output = pd.DataFrame({'Row_labels_1': ['uni vs prof_qual', 'uni vs prof_qual',
                                                      'uni vs a_level', 'uni vs a_level',
                                                      'uni vs gcse', 'uni vs gcse',
                                                      'prof_qual vs a_level', 'prof_qual vs a_level',
                                                      'prof_qual vs gcse', 'prof_qual vs gcse',
                                                      'a_level vs gcse', 'a_level vs gcse'],
                                     'Row_labels_2': ['p_val', 'rho',
                                                      'p_val', 'rho',
                                                      'p_val', 'rho',
                                                      'p_val', 'rho',
                                                      'p_val', 'rho',
                                                      'p_val', 'rho']})
    education_output.set_index('Row_labels_1', 'Row_labels_2')

    dataset_uni = dataset.groupby('Education_highest').get_group(UNIVERSITY_CODE)
    dataset_prof_qual = dataset.groupby('Education_highest').get_group(PROFESSIONAL_QUAL_CODE)
    dataset_a_level = dataset.groupby('Education_highest').get_group(A_LEVEL_CODE)
    dataset_gcse = dataset.groupby('Education_highest').get_group(GCSE_CODE)

    # Independent t-tests with alpha corrected for multiple comparisons using Bonferroni's method
    alpha_corrected = 0.05 / 6

    for y in y_list:
        y_results = []

        print('\n', y)
        tstat, pval = ttest_ind(dataset_uni[y], dataset_prof_qual[y])
        effect_size = cohend(dataset_uni[y], dataset_prof_qual[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"uni vs prof_qual [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        tstat, pval = ttest_ind(dataset_uni[y], dataset_a_level[y])
        effect_size = cohend(dataset_uni[y], dataset_a_level[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"uni vs a_level [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        tstat, pval = ttest_ind(dataset_uni[y], dataset_gcse[y])
        effect_size = cohend(dataset_uni[y], dataset_gcse[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"uni vs gcse [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        tstat, pval = ttest_ind(dataset_prof_qual[y], dataset_a_level[y])
        effect_size = cohend(dataset_prof_qual[y], dataset_a_level[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"prof_qual vs a_level [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        tstat, pval = ttest_ind(dataset_prof_qual[y], dataset_gcse[y])
        effect_size = cohend(dataset_prof_qual[y], dataset_gcse[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"prof_qual vs gcse [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        tstat, pval = ttest_ind(dataset_a_level[y], dataset_gcse[y])
        effect_size = cohend(dataset_a_level[y], dataset_gcse[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"a_level vs gcse [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        education_output[y] = y_results

    education_output.to_csv(correlation_dir / f'education_ttest_{model_name}output.csv')


if __name__ == '__main__':
    main(args.experiment_name,
         args.model_name)
