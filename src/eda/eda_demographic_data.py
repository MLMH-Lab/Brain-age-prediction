#!/usr/bin/env python3
"""Exploratory data analysis.

This script records the frequency of different demographic features, the age histogram for both genders, and the
description of the whole dataset and the genders groups.

"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from utils import load_demographic_data

plt.style.use('seaborn-whitegrid')
PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='homogenized_ids.csv',
                    help='Filename indicating the ids to be used.')

parser.add_argument('-U', '--suffix',
                    dest='suffix',
                    help='Name of the suffix.')

args = parser.parse_args()


def save_frequency_table(output_dir, input_df, col_name, suffix):
    """Export frequency table of column as csv."""
    freq_table = input_df[col_name].value_counts()
    print(col_name)
    print(freq_table)
    freq_table.to_csv(output_dir / f'{col_name}{suffix}_freq.csv')


def create_histogram(output_dir, input_df, suffix):
    """Save histogram of age distribution by gender in experiment directory as eps."""
    plt.figure(figsize=(10, 7))

    plt.hist(input_df.Age, color='blue', histtype='step', lw=2, bins=input_df.Age.nunique())

    plt.title('Age distribution in UK BIOBANK', fontsize=17)
    plt.xlabel('Age at MRI scan [years]', fontsize=15)
    plt.ylabel('Number of subjects', fontsize=15)
    plt.tick_params(labelsize=13)
    plt.axis('tight')

    plt.savefig(output_dir / f'age_dist{suffix}.eps', format='eps')


def create_gender_histogram(output_dir, input_df, suffix):
    """Save histogram of age distribution by gender in experiment directory as eps."""
    male_code = 1
    female_code = 0

    male_ages = input_df.groupby('Gender').get_group(male_code).Age
    female_ages = input_df.groupby('Gender').get_group(female_code).Age

    plt.figure(figsize=(10, 7))

    plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=male_ages.nunique(), label='male')
    plt.hist(female_ages, color='red', histtype='step', lw=2, bins=female_ages.nunique(), label='female')

    plt.title('Age distribution in UK BIOBANK by sex', fontsize=17)
    plt.xlabel('Age at MRI scan [years]', fontsize=15)
    plt.ylabel('Number of subjects', fontsize=15)
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)
    plt.axis('tight')

    plt.savefig(output_dir / f'sex_age_dist{suffix}.eps', format='eps')


def main(experiment_name, scanner_name, input_ids_file, suffix):
    """Perform the exploratory data analysis."""
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    ids_path = experiment_dir / input_ids_file

    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    eda_dir = experiment_dir / 'EDA'
    eda_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(participants_path, ids_path)

    # Export ethnicity and age distribution for future reference
    save_frequency_table(eda_dir, dataset, 'Ethnicity', suffix)
    save_frequency_table(eda_dir, dataset, 'Age', suffix)
    save_frequency_table(eda_dir, dataset, 'Gender', suffix)

    create_gender_histogram(eda_dir, dataset, suffix)
    create_histogram(eda_dir, dataset, suffix)

    print('Whole dataset')
    print(dataset.Age.describe())
    dataset.Age.describe().to_csv(eda_dir / f'whole_dataset_dem{suffix}.csv')

    print('Grouped dataset')
    print(dataset.groupby('Gender').Age.describe())
    dataset.groupby('Gender').Age.describe().to_csv(eda_dir / f'grouped_dem{suffix}.csv')


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file, args.suffix)
