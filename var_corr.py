"""Script to assess correlations between BrainAGE/BrainAGER and demographic variables in UK Biobank
(dataset created in variables_biobank.py)"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, f_oneway, ttest_ind
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

uni_code = 4
prof_qual_code = 3
a_level_code = 2
gcse_code = 1


def spearman(df, x, y):
    """Calculate and interpret spearman's correlation of cols x and y"""

    spearman_rho, spearman_p = spearmanr(df[x], df[y])

    alpha = 0.05
    n = len(df)
    if spearman_p < alpha:
        print('n=%s, %s and %s - reject H0: p = %.3f, rho = %.3f'
              % (n, x, y, spearman_p, spearman_rho))


def cohend(d1, d2):
    """Function to calculate Cohen's d for independent samples"""

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    effect_size = (u1 - u2) / s
    print(d1, d2, effect_size, '\n')


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

    # Spearman correlation per variable
    print("Spearman correlation")
    for y in y_list:
        for x in x_list:
            dataset_y = dataset.dropna(subset=[y])
            spearman(dataset_y, x, y)


    dataset_uni = dataset.groupby('Education_highest').get_group(uni_code)
    dataset_prof_qual = dataset.groupby('Education_highest').get_group(prof_qual_code)
    dataset_a_level = dataset.groupby('Education_highest').get_group(a_level_code)
    dataset_gcse = dataset.groupby('Education_highest').get_group(gcse_code)


    # bonferroni-corrected alpha for multiple t-tests
    alpha_bon = 0.05 / 6

    for x in x_list:
        plist = []
        print('\n', x)
        tstat, pval = ttest_ind(dataset_uni[x], dataset_prof_qual[x])
        plist.append(pval)
        if pval < alpha_bon:
            print("uni vs prof_qual", pval)
        tstat, pval = ttest_ind(dataset_uni[x], dataset_a_level[x])
        plist.append(pval)
        if pval < alpha_bon:
            print("uni vs a_level", pval)
        tstat, pval = ttest_ind(dataset_uni[x], dataset_gcse[x])
        plist.append(pval)
        if pval < alpha_bon:
            print("uni vs gcse", pval)
        tstat, pval = ttest_ind(dataset_prof_qual[x], dataset_a_level[x])
        plist.append(pval)
        if pval < alpha_bon:
            print("prof_qual vs a_level", pval)
        tstat, pval = ttest_ind(dataset_prof_qual[x], dataset_gcse[x])
        plist.append(pval)
        if pval < alpha_bon:
            print("prof_qual vs gcse", pval)
        tstat, pval = ttest_ind(dataset_a_level[x], dataset_gcse[x])
        plist.append(pval)
        if pval < alpha_bon:
            print("a_level vs gcse", pval)
        # print(multipletests(plist, alpha=0.05, method='bonferroni'))

    # Cohen's d test for education levels # outputs for all vars
    cohend(dataset_uni, dataset_prof_qual)
    cohend(dataset_uni, dataset_a_level)
    cohend(dataset_uni, dataset_gcse)
    cohend(dataset_prof_qual, dataset_a_level)
    cohend(dataset_prof_qual, dataset_gcse)
    cohend(dataset_a_level, dataset_gcse)


if __name__ == "__main__":
    main()
