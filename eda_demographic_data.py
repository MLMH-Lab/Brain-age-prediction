"""Exploratory data analysis.

This script records the frequency of different demographic features, the age histogram for both genders, and the
description of the whole dataset and the genders groups.

"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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

    # Loading supplementary demographic data
    demographic_df = pd.read_csv(demographic_path, usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    demographic_df.columns = ['ID', 'Gender', 'Ethnicity', 'Age']
    demographic_df = demographic_df.dropna()

    id_df = pd.read_csv(id_path)
    # Create a new 'ID' column to match supplementary demographic data
    if 'Participant_ID' in id_df.columns:
        # For create_homogeneous_data.py output
        id_df['ID'] = id_df['Image_ID'].str.split('-').str[1]
    else:
        # For freesurferData dataframe
        id_df['ID'] = id_df['Image_ID'].str.split('_').str[0]
        id_df['ID'] = id_df['ID'].str.split('-').str[1]

    id_df['ID'] = pd.to_numeric(id_df['ID'])

    # Merge supplementary demographic data with ids
    dataset = pd.merge(id_df, demographic_df, on='ID')

    # Labeling data
    ethnicity_dict = {
        1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
        2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
        3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
        4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
        5: 'Chinese',
        6: 'Other',
        -1: 'Not known', -3: 'Not known'
    }

    gender_dict = {
        0: 'Female',
        1: 'Male'
    }

    dataset = dataset.replace({'Gender': gender_dict})
    dataset = dataset.replace({'Ethnicity': ethnicity_dict})

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
