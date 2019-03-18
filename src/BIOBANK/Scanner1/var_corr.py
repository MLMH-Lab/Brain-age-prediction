"""Script to assess correlations between difference in actual and
predicted age with demographic variables in UK BIOBANK Scanner1

Variables to assess [variable code - variable name, code names (where applicable)]:
6138-2.0, 6138-2.1, 6138-2.2, 6138-2.3, 6138-2.4 - Which of the following qualifications do you have? (up to 5 selections possible)
1	College or University degree
2	A levels/AS levels or equivalent
3	O levels/GCSEs or equivalent
4	CSEs or equivalent
5	NVQ or HND or HNC or equivalent
6	Other professional qualifications eg: nursing, teaching
-7	None of the above
-3	Prefer not to answer
Researcher notes:
- CSE is the predecessor of GCSE, so can be treated the same
- NVQ/HND/HNC are work-based qualifications/degrees, comparable to short undergraduate degrees

24005-0.0 - Particulate matter air pollution (pm10); 2010
24009-0.0 - Traffic intensity on the nearest road
24010-0.0 - Inverse distance to the nearest road
24014-0.0 - Close to major road (binary)
24500-0.0 - Greenspace percentage
24501-0.0 - Domestic garden percentage
24502-0.0 - Water percentage
24506-0.0 - Natural environment percentage

Note: Baseline assessment data chosen for 24500-24506 because data is not available for all subjects at next assessment;
no data available for imaging assessment

Variable information available at https://biobank.ctsu.ox.ac.uk/crystal/label.cgi;
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, f_oneway, ttest_ind
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib.pyplot as plt


PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def spearman(df, x, y):
    """Calculate and interpret spearman's correlation of cols x and y"""

    spearman_rho, spearman_p = spearmanr(df[x], df[y])

    alpha = 0.05
    n = len(df)
    if spearman_p < alpha:
        print('n=%s, %s and %s - reject H0: p = %.3f, rho = %.3f'
              % (n, x, y, spearman_p, spearman_rho))
    # elif spearman_p >= alpha:
    #     print('%s and %s are uncorrelated (fail to reject H0): p = %.3f, rho = %.3f'
    #           % (x, y, spearman_p, spearman_rho))
    # else:
    #     print('Error with %s and %s' % (x, y))


def ols_reg(df, indep, dep):
    """Perform linear regression using ordinary least squares (OLS) method"""

    endog = np.asarray(df[indep], dtype=float)
    exog = np.asarray(sm.add_constant(df[dep]), dtype=float)
    OLS_model = sm.OLS(endog, exog)
    OLS_results = OLS_model.fit()
    OLS_p = OLS_results.pvalues[1]
    OLS_coeff = OLS_results.params[1]

    alpha = 0.05
    n = len(df)
    if OLS_p < alpha:
        print('n=%s, %s and %s - reject H0: p = %.3f, coef = %.3f'
              % (n, indep, dep, OLS_p, OLS_coeff))
    # elif OLS_p >= alpha:
    #     print('%s and %s - fail to reject H0: p = %.3f, coef = %.3f'
    #           % (indep, dep, OLS_p, OLS_coeff))
    # else:
    #     print('Error with %s and %s' % (indep, dep))


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
    # Define what subjects dataset should contain: total, male or female
    subjects = 'total'

    # Load dataset with age vars, demographic data from Biobank, demographic data from IMD
    dataset = pd.read_csv(str(PROJECT_ROOT / 'outputs' / subjects / 'age_predictions_demographics.csv'))

    # Correlation variables
    x_list = ['Abs_BrainAGE_predmean', 'Abs_BrainAGE_predmedian',
              'Abs_BrainAGER_predmean', 'Abs_BrainAGER_predmedian',
              'BrainAGE_predmean', 'BrainAGE_predmedian',
              'BrainAGER_predmean', 'BrainAGER_predmedian',
              'Std_predicted_age']
    y_list = ['Education_highest',
              'Air_pollution',
              'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
              'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Spearman correlation per variable
    print("Spearman correlation")
    for x in x_list:
        for y in y_list:
            dataset_y = dataset.dropna(subset=[y])
            spearman(dataset_y, x, y)

    # Linear regression per variable
    print("\n OLS regression")
    for x in x_list:
        for y in y_list:
            dataset_y = dataset.dropna(subset=[y])
            ols_reg(dataset_y, x, y)

    # Scatterplots
    air_pollution_plot = dataset.plot(x='BrainAGER_predmean', y='Air_pollution', kind='scatter')
    traffic_intensity_plot = dataset.plot(x='BrainAGER_predmean', y='Traffic_intensity', kind='scatter')
    inv_dist_plot = dataset.plot(x='BrainAGER_predmean', y='Inverse_dist_road', kind='scatter')
    greenspace_plot = dataset.plot(x='BrainAGER_predmean', y='Greenspace_perc', kind='scatter')
    garden_plot = dataset.plot(x='BrainAGER_predmean', y='Garden_perc', kind='scatter')
    water_plot = dataset.plot(x='BrainAGER_predmean', y='Water_perc', kind='scatter')
    nat_env_plot = dataset.plot(x='BrainAGER_predmean', y='Natural_env_perc', kind='scatter')

    # Exploratory analysis of education
    education_fre = pd.crosstab(index=dataset["Education_highest"], columns="count")

    # Perform one-way ANOVA for education groups
    # Alternative to: ols_reg(dataset_y, 'Diff age-mean', 'Education_highest') ?
    uni_code = 4
    prof_qual_code = 3
    a_level_code = 2
    gcse_code = 1

    dataset_uni = dataset.groupby('Education_highest').get_group(uni_code)
    dataset_prof_qual = dataset.groupby('Education_highest').get_group(prof_qual_code)
    dataset_a_level = dataset.groupby('Education_highest').get_group(a_level_code)
    dataset_gcse = dataset.groupby('Education_highest').get_group(gcse_code)

    for x in x_list:
        f_stat, pvalue = f_oneway(dataset_uni[x], dataset_prof_qual[x], dataset_a_level[x], dataset_gcse[x])
        print(x, f_stat, pvalue)

    # Boxplot for education
    df_edu = pd.concat([dataset_uni['Abs_BrainAGER_predmean'], dataset_prof_qual['Abs_BrainAGER_predmean'],
                        dataset_a_level['Abs_BrainAGER_predmean'], dataset_gcse['Abs_BrainAGER_predmean']],
                        axis=1, keys=['Uni', 'Prof_qual', 'A_levels', 'GCSE'])

    plot = pd.DataFrame.boxplot(df_edu)

    output_img_path = PROJECT_ROOT/'outputs'/'edu_abs_agemean_BIOBANK.png'
    plt.savefig(str(output_img_path))

    # Holm-Bonferroni method for multiple comparisons
    dataset = dataset.dropna(subset=['Education_highest'])
    mod = MultiComparison(dataset['Abs_BrainAGER_predmean'], dataset['Education_highest'])
    print(mod.tukeyhsd())

    # bonferroni-corrected alpha for multiple t-tests
    alpha_bon = 0.05/6

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
        print(multipletests(plist, alpha=0.05, method='bonferroni'))

    # Cohen's d test for education levels # outputs for all vars
    cohend(dataset_uni, dataset_prof_qual)
    cohend(dataset_uni, dataset_a_level)
    cohend(dataset_uni, dataset_gcse)
    cohend(dataset_prof_qual, dataset_a_level)
    cohend(dataset_prof_qual, dataset_gcse)
    cohend(dataset_a_level, dataset_gcse)


if __name__ == "__main__":
    main()
