#!/usr/bin/env python3
"""Script to plot distribution of education with regards to age in UK Biobank Scanner1 dataset"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-whitegrid')
PROJECT_ROOT = Path.cwd()


def plot_save_histogram_education(output_dir, input_df, suffix):
    """Create histogram of age distribution by education level and save in output folder as eps"""
    gcse_code = 1
    alevels_code = 2
    prof_qual_code = 3
    uni_code = 4

    gcse_ages = input_df.groupby('Education_highest').get_group(gcse_code).Age
    alevels_ages = input_df.groupby('Education_highest').get_group(alevels_code).Age
    prof_qual_ages = input_df.groupby('Education_highest').get_group(prof_qual_code).Age
    uni_ages = input_df.groupby('Education_highest').get_group(uni_code).Age

    plt.figure(figsize=(10, 7))

    plt.hist(gcse_ages, color='blue', histtype='step', lw=2, bins=gcse_ages.nunique(), label='GCSE')
    plt.hist(alevels_ages, color='red', histtype='step', lw=2, bins=alevels_ages.nunique(), label='A levels')
    plt.hist(prof_qual_ages, color='yellow', histtype='step', lw=2, bins=prof_qual_ages.nunique(),
             label='Professional qualification')
    plt.hist(uni_ages, color='green', histtype='step', lw=2, bins=uni_ages.nunique(), label='University')

    plt.title('Age distribution in UK BIOBANK by education level', fontsize=17)
    plt.xlabel('Age at MRI scan [years]', fontsize=15)
    plt.ylabel('Number of subjects', fontsize=15)
    plt.legend(loc='upper left', fontsize=13)
    plt.tick_params(labelsize=13)
    plt.axis('tight')

    plt.savefig(output_dir / f'education_age_dist{suffix}.eps', format='eps')


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    suffix = '_all'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    eda_dir = experiment_dir / 'EDA'
    eda_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------------------
    correlation_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'correlation_analysis'

    ensemble_df = pd.read_csv(correlation_dir / 'ensemble_SVM_output.csv')
    variables_df = pd.read_csv(PROJECT_ROOT / 'outputs' / 'covariates' / 'covariates.csv')

    ensemble_df['id'] = ensemble_df['image_id'].str.split('-').str[1]
    ensemble_df['id'] = ensemble_df['id'].str.split('_').str[0]
    ensemble_df['id'] = pd.to_numeric(ensemble_df['id'])

    dataset = pd.merge(ensemble_df, variables_df, on='id')
    dataset = dataset.dropna(subset=['Education_highest'])

    plot_save_histogram_education(eda_dir, dataset, suffix)


if __name__ == '__main__':
    main()
