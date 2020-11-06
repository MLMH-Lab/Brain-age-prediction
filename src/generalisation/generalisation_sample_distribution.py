"""Script to explore age and MAE distribution in UK Biobank site 2;

All models performed better in the independent test set (site 2), so this
script explores whether the age distribution in site 2 may be skewed towards
ages that had lower MAE in the training set (site 1)"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import ttest_ind
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    experiment_name = 'biobank_scanner2'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # -------------------------------------------
    # Load file with all model predictions (averaged across iterations)
    # and get included ages
    age_predictions_all = pd.read_csv(
        experiment_dir / 'age_predictions_test_allmodels.csv', index_col=0)
    ages = age_predictions_all['Age'].unique()
    ages_ls = ages.tolist()
    ages_ls.sort()

    # Get total sample size
    n_total = len(age_predictions_all)

    # Set up df to store results incl number of subjects per age and
    # % of total sample per age group
    results = pd.DataFrame(columns=['Age', 'n', 'n_percentage'])

    n_per_age_ls = []
    n_perc_ls = []

    for age in ages_ls:
        subjects_per_age = age_predictions_all.groupby('Age').get_group(age)
        n_per_age = len(subjects_per_age)
        n_perc = n_per_age / n_total * 100

        n_per_age_ls.append(n_per_age)
        n_perc_ls.append(n_perc)

    results['Age'] = ages_ls
    results['n'] = n_per_age_ls
    results['n_percentage'] = n_perc_ls

    # Loop over models and ages to obtain MAE per age
    for model_name in model_ls:
        model_mae_ls = []

        for age in ages_ls:
            subjects_per_age = age_predictions_all.groupby('Age').get_group(age)

            mae_per_age = mean_absolute_error(subjects_per_age['Age'],
                                               subjects_per_age[model_name])
            model_mae_ls.append(mae_per_age)

        results[model_name] = model_mae_ls

    # Save results
    results.to_csv(experiment_dir / 'test_mae_per_age.csv', index=False)

    # ------------------------
    # Compare age mean in sites 1 and 2
    site2_ages = age_predictions_all['Age']

    age_predictions_site1 = pd.read_csv(PROJECT_ROOT / 'outputs' /
                                        'biobank_scanner1' /
                                        'age_predictions_allmodels.csv',
                                        index_col=0)
    site1_ages = age_predictions_site1['Age']
    tstat, pval = ttest_ind(site1_ages, site2_ages)


if __name__ == '__main__':
    main()