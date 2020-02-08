#!/usr/bin/env python3
"""Script to create files with subjects' ids to perform sample size analysis

This script creates gender-homogeneous bootstraped datasets.
Creates 20 bootstrap samples with increasing size
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_demographic_data

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

parser.add_argument('-N', '--n_bootstrap',
                    dest='n_bootstrap',
                    type=int, default=1000,
                    help='Number of bootstrap iterations.')

parser.add_argument('-R', '--n_max_pair',
                    dest='n_max_pair',
                    type=int, default=20,
                    help='Number maximum of pairs.')

args = parser.parse_args()


def main(experiment_name, scanner_name, input_ids_file, n_bootstrap, n_max_pair):
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    ids_path = experiment_dir / input_ids_file

    sample_size_dir = experiment_dir / 'sample_size'
    sample_size_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------------------
    # Set random seed for random sampling of subjects
    np.random.seed(42)

    dataset = load_demographic_data(participants_path, ids_path)

    # Find range of ages in homogeneous dataset
    age_min = int(dataset['Age'].min())  # 47
    age_max = int(dataset['Age'].max())  # 73

    # Loop to create 20 bootstrap samples that each contain up to 20 gender-balanced subject pairs per age group/year
    # Create a out-of-bag set (~test set)
    for i_n_subject_pairs in range(1, n_max_pair + 1):
        print(i_n_subject_pairs)
        ids_with_n_subject_pairs_dir = sample_size_dir / f'{i_n_subject_pairs:02d}'
        ids_with_n_subject_pairs_dir.mkdir(exist_ok=True)
        ids_dir = ids_with_n_subject_pairs_dir / 'ids'
        ids_dir.mkdir(exist_ok=True)

        # Loop to create 1000 random subject samples of the same size (with replacement) per bootstrap sample
        for i_bootstrap in range(n_bootstrap):
            # Create empty df to add bootstrap subjects to
            dataset_bootstrap_train = pd.DataFrame(columns=['image_id'])
            dataset_bootstrap_test = pd.DataFrame(columns=['image_id'])

            # Loop over ages (27 in total)
            for age in range(age_min, (age_max + 1)):

                # Get dataset for specific age only
                age_group = dataset.groupby('Age').get_group(age)

                # Loop over genders (0: female, 1:male)
                for gender in range(2):
                    gender_group = age_group.groupby('Gender').get_group(gender)

                    # Extract random subject of that gender and add to dataset_bootstrap_train
                    random_sample_train = gender_group.sample(n=i_n_subject_pairs, replace=True)
                    dataset_bootstrap_train = pd.concat([dataset_bootstrap_train, random_sample_train[['image_id']]])

                    # Sample test set with always the same size
                    not_sampled = ~gender_group['image_id'].isin(random_sample_train['image_id'])
                    random_sample_test = gender_group[not_sampled].sample(n=20, replace=False)
                    dataset_bootstrap_test = pd.concat([dataset_bootstrap_test, random_sample_test[['image_id']]])

            # Export dataset_bootstrap_train as csv
            output_prefix = f'{i_bootstrap:04d}_{i_n_subject_pairs:02d}'
            dataset_bootstrap_train.to_csv(ids_dir / f'{output_prefix}_train.csv', index=False)
            dataset_bootstrap_test.to_csv(ids_dir / f'{output_prefix}_test.csv', index=False)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file,
         args.n_bootstrap, args.n_max_pair)
