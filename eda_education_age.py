#!/usr/bin/env python3
"""Script to plots distribution of education with regards to age in UK Biobank Scanner1 dataset"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path.cwd()


def plot_save_histogram_education(edu_level_1, edu_level_2, edu_level_3, edu_level_4):
    """Create histogram of age distribution by education level and saves in output folder as png"""
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 7))

    plt.hist(edu_level_1, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='GCSE')
    plt.hist(edu_level_2, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='A levels')
    plt.hist(edu_level_3, color='yellow', histtype='step', lw=2, bins=range(45, 75, 1),
             label='Professional qualification')
    plt.hist(edu_level_4, color='green', histtype='step', lw=2, bins=range(45, 75, 1), label='University')

    plt.title("Age distribution in UK BIOBANK by education level", fontsize=17)
    plt.axis('tight')
    plt.xlabel("Age at MRI scan [years]", fontsize=15)
    plt.ylabel("Number of subjects", fontsize=15)
    plt.xticks(range(45, 75, 5))
    plt.yticks(range(0, 401, 50))
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)

    output_img_path = PROJECT_ROOT / 'outputs' / 'education_age_dist_BIOBANK.png'
    plt.savefig(str(output_img_path))
    plt.show()


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
    dataset = dataset.dropna(subset=['Education_highest'])

    # Histogram of age distribution by education level
    gcse_code = 1
    alevels_code = 2
    prof_qual_code = 3
    uni_code = 4

    gcse_ages = dataset.groupby('Education_highest').get_group(gcse_code).Age
    alevels_ages = dataset.groupby('Education_highest').get_group(alevels_code).Age
    prof_qual_ages = dataset.groupby('Education_highest').get_group(prof_qual_code).Age
    uni_ages = dataset.groupby('Education_highest').get_group(uni_code).Age

    plot_save_histogram_education(gcse_ages, alevels_ages, prof_qual_ages, uni_ages)


if __name__ == "__main__":
    main()
