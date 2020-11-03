"""Script to explore age and MAE distribution in UK Biobank site 2;

All models performed better in the independent test set (site 2), so this
script explores whether the age distribution in site 2 may be skewed towards
ages that had lower MAE in the training set (site 1)"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    experiment_name = 'biobank_scanner2'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # -------------------------------------------
    # Get included ages from example model file
    age_predictions_all = pd.read_csv(
        experiment_dir / 'age_predictions_test_allmodels.csv')
    ages = age_predictions_all['Age'].unique()
    ages_ls = ages.tolist()
    ages_ls.sort()

    # Get total sample size
    n_total = len(age_predictions_all)

    # Loop over models and ages to obtain MAE per age
    for model_name in model_ls:
        model_dir = experiment_dir / model_name

        results = pd.DataFrame(columns=['Age', 'n',
                                        'n_percentage', 'mae_per_age'])

        # Loop to calculate statistics per age group: sample size per age,
        # % of total sample in age group, MAE per age group
        for age in ages_ls:
            subjects_per_age = age_predictions_all.groupby('Age').get_group(age)

            n_per_age = len(subjects_per_age)
            n_perc = n_per_age / n_total * 100

            mae_per_age = mean_absolute_error(subjects_per_age['Age'],
                                               subjects_per_age[model_name])

            print(model_name, age, n_per_age, n_perc, mae_per_age)

        results = results.append(
            {'Age': age, 'n': n_per_age,
             'n_percentage': n_perc, 'mae_per_age': mae_per_age},
            ignore_index=True)

        results.to_csv(model_dir / f'{model_name}_test_mae_per_age.csv',
                       index=False)

    # ----------------------------------------
    # Check at what age 50% of subjects are included
    for model_name in model_ls:
        percentage = 0

        for age in ages_ls:
            subjects_per_age = age_predictions_all.groupby('Age').get_group(age)
            n_per_age = len(subjects_per_age)
            n_perc = n_per_age / n_total * 100
            percentage += n_perc
            if percentage >= 75 and percentage <= 85:
                print(model_name, age, percentage)
        print('')


if __name__ == '__main__':
    main()