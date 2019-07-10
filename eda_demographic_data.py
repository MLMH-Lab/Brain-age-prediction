"""Exploratory data analysis.

This script records the frequency of different demographic features, the age histogram for both genders, and the
description of the whole dataset and the genders groups.

"""
from pathlib import Path

import matplotlib.pyplot as plt

from helper_functions import load_demographic_data

plt.style.use('seaborn-whitegrid')
PROJECT_ROOT = Path.cwd()


def save_frequency_table(output_dir, input_df, col_name, suffix):
    """Export frequency table of column as csv."""
    freq_table = input_df[col_name].value_counts()
    print(col_name)
    print(freq_table)
    file_name = col_name + suffix + '_freq.csv'
    freq_table.to_csv(output_dir / file_name)


def create_gender_histogram(output_dir, input_df, suffix):
    """Save histogram of age distribution by gender in experiment directory as png."""

    male_ages = input_df.groupby('Gender').get_group('Male').Age
    female_ages = input_df.groupby('Gender').get_group('Female').Age

    plt.figure(figsize=(10, 7))

    plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='male')
    plt.hist(female_ages, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='female')

    plt.title("Age distribution by gender", fontsize=17)
    plt.xlabel("Age at MRI scan [years]", fontsize=15)
    plt.ylabel("Number of subjects", fontsize=15)
    plt.xticks(range(45, 75, 5))
    plt.yticks(range(0, 401, 50))
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)
    plt.axis('tight')

    file_name = 'gender_age_dist' + suffix + '.png'

    plt.savefig(str(output_dir / file_name))


def main():
    """Perform the exploratory data analysis."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    suffix_analysis_phase = '_initial_analysis'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    id_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'
    # ----------------------------------------------------------------------------------------

    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    eda_dir = experiment_dir / 'EDA'
    eda_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(demographic_path, id_path)

    # Export ethnicity and age distribution for future reference
    save_frequency_table(eda_dir, dataset, 'Ethnicity', suffix_analysis_phase)
    save_frequency_table(eda_dir, dataset, 'Age', suffix_analysis_phase)
    save_frequency_table(eda_dir, dataset, 'Gender', suffix_analysis_phase)

    create_gender_histogram(eda_dir, dataset, suffix_analysis_phase)

    print('Whole dataset')
    print(dataset.Age.describe())
    dataset.Age.describe().to_csv(eda_dir / ('whole_dataset_dem' + suffix_analysis_phase + '.csv'))

    print('Grouped dataset')
    print(dataset.groupby('Gender').Age.describe())
    dataset.groupby('Gender').Age.describe().to_csv(eda_dir / ('grouped_dem' + suffix_analysis_phase + '.csv'))


if __name__ == "__main__":
    main()
