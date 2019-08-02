"""Script to assess correlations between BrainAGE/BrainAGER and demographic variables in UK Biobank
(dataset created in variables_biobank.py)"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_ind

PROJECT_ROOT = Path.cwd()

uni_code = 4
prof_qual_code = 3
a_level_code = 2
gcse_code = 1


def spearman(df, x, y):
    """Calculate and interpret Spearman's rank-order correlation of variables x and y"""

    spearman_rho, spearman_p = spearmanr(df[x], df[y])

    alpha = 0.05
    n = len(df)

    if spearman_p < alpha:
        print('n=%s, %s and %s - reject H0: p = %.3f, rho = %.3f'
              % (n, x, y, spearman_p, spearman_rho))
    else:
        print('n=%s, %s and %s - not significant: p = %.3f, rho = %.3f'
              % (n, x, y, spearman_p, spearman_rho))


def cohend(d1, d2):
    """Calculate Cohen's d effect size for independent samples"""

    n1, n2 = len(d1), len(d2)
    d1_variance, d2_variance = np.var(d1, ddof=1), np.var(d2, ddof=1)
    std_pooled = np.sqrt(((n1 - 1) * d1_variance + (n2 - 1) * d2_variance) / (n1 + n2 - 2))
    d1_mean, d2_mean = np.mean(d1), np.mean(d2)
    effect_size = (d1_mean - d2_mean) / std_pooled

    return effect_size


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    # ----------------------------------------------------------------------------------------
    correlation_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'correlation_analysis'

    ensemble_df = pd.read_csv(correlation_dir / 'ensemble_output.csv')
    ensemble_df['ID'] = ensemble_df['Participant_ID'].str.split('-').str[1]
    ensemble_df['ID'] = pd.to_numeric(ensemble_df['ID'])

    variables_df = pd.read_csv(correlation_dir / 'variables_biobank.csv')

    dataset = pd.merge(ensemble_df, variables_df, on='ID')

    # Correlation variables
    y_list = ['BrainAGE_predmean', 'BrainAGE_predmedian',
              'BrainAGER_predmean', 'BrainAGER_predmedian',
              'Std_predicted_age']
    x_list = ['Air_pollution',
              'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
              'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Spearman's correlation per variable
    print("Spearman's correlation")
    for y in y_list:
        for x in x_list:
            dataset_x = dataset.dropna(subset=[x])
            print(y, x, len(dataset_x))
            spearman(dataset_x, x, y)

    # Analysis of education level
    dataset_uni = dataset.groupby('Education_highest').get_group(uni_code)
    dataset_prof_qual = dataset.groupby('Education_highest').get_group(prof_qual_code)
    dataset_a_level = dataset.groupby('Education_highest').get_group(a_level_code)
    dataset_gcse = dataset.groupby('Education_highest').get_group(gcse_code)

    # Independent t-tests with alpha corrected for multiple comparisons using Bonferroni's method
    alpha_corrected = 0.05 / 6

    for y in y_list:

        print('\n', y)
        tstat, pval = ttest_ind(dataset_uni[y], dataset_prof_qual[y])
        effect_size = cohend(dataset_uni[y], dataset_prof_qual[y])
        if pval < alpha_corrected:
            print("uni vs prof_qual [t-test pval, cohen's d]", pval, effect_size)

        tstat, pval = ttest_ind(dataset_uni[y], dataset_a_level[y])
        effect_size = cohend(dataset_uni[y], dataset_a_level[y])
        if pval < alpha_corrected:
            print("uni vs a_level [t-test pval, cohen's d]", pval, effect_size)

        tstat, pval = ttest_ind(dataset_uni[y], dataset_gcse[y])
        effect_size = cohend(dataset_uni[y], dataset_gcse[y])
        if pval < alpha_corrected:
            print("uni vs gcse [t-test pval, cohen's d]", pval, effect_size)

        tstat, pval = ttest_ind(dataset_prof_qual[y], dataset_a_level[y])
        effect_size = cohend(dataset_prof_qual[y], dataset_a_level[y])
        if pval < alpha_corrected:
            print("prof_qual vs a_level [t-test pval, cohen's d]", pval, effect_size)

        tstat, pval = ttest_ind(dataset_prof_qual[y], dataset_gcse[y])
        effect_size = cohend(dataset_prof_qual[y], dataset_gcse[y])
        if pval < alpha_corrected:
            print("prof_qual vs gcse [t-test pval, cohen's d]", pval, effect_size)

        tstat, pval = ttest_ind(dataset_a_level[y], dataset_gcse[y])
        effect_size = cohend(dataset_a_level[y], dataset_gcse[y])
        if pval < alpha_corrected:
            print("a_level vs gcse [t-test pval, cohen's d]", pval, effect_size)


if __name__ == "__main__":
    main()
