#!/usr/bin/env python3
"""
Script to create gender-homogeneous bootstraped datasets to feed into create_h5_bootstrap script;
Creates 20 bootstrap samples with increasing size
"""
from pathlib import Path

import pandas as pd
import numpy as np

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'dataset_homogeneous.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    bootstrap_dir = experiment_dir / 'bootstrap_analysis'
    bootstrap_dir.mkdir(exist_ok=True)

    # Set random seed for random sampling of subjects
    np.random.seed(42)

    dataset = load_demographic_data(demographic_path, id_path)

    # Find range of ages in homogeneous dataset
    age_min = int(dataset['Age'].min())  # 47
    age_max = int(dataset['Age'].max())  # 73

    # Number maximum of pairs
    n_max_pair = 20
    # Loop to create 20 bootstrap samples that each contain up to 20 gender-balanced subject pairs per age group/year
    # Create a out-of-bag set (~test set)
    for i_n_subject_pairs in range(1, n_max_pair + 1):
        print(i_n_subject_pairs)
        ids_with_n_subject_pairs_dir = bootstrap_dir / '{:02d}'.format(i_n_subject_pairs)
        ids_with_n_subject_pairs_dir.mkdir(exist_ok=True)
        ids_dir = ids_with_n_subject_pairs_dir / 'ids'
        ids_dir.mkdir(exist_ok=True)

        # Loop to create 1000 random subject samples of the same size (with replacement) per bootstrap sample
        n_bootstrap = 1000
        for i_bootstrap in range(n_bootstrap):
            # Create empty df to add bootstrap subjects to
            dataset_bootstrap_train = pd.DataFrame(columns=['Participant_ID'])
            dataset_bootstrap_test = pd.DataFrame(columns=['Participant_ID'])

            # Loop over ages (27 in total)
            for age in range(age_min, (age_max + 1)):

                # Get dataset for specific age only
                age_group = dataset.groupby('Age').get_group(age)

                # Loop over genders (0: female, 1:male)
                for gender in range(2):
                    gender_group = age_group.groupby('Gender').get_group(gender)

                    # Extract random subject of that gender and add to dataset_bootstrap_train
                    random_sample_train = gender_group.sample(n=i_n_subject_pairs, replace=True)
                    dataset_bootstrap_train = pd.concat(
                        [dataset_bootstrap_train, pd.DataFrame(random_sample_train['Participant_ID'])])

                    # Sample test set with always the same size
                    not_sampled = ~gender_group['Participant_ID'].isin(random_sample_train['Participant_ID'])
                    random_sample_test = gender_group[not_sampled].sample(n=n_max_pair, replace=False)
                    dataset_bootstrap_test = pd.concat(
                        [dataset_bootstrap_test, pd.DataFrame(random_sample_test['Participant_ID'])])

            # Export dataset_bootstrap_train as csv
            ids_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}_train.csv'.format(i_bootstrap, i_n_subject_pairs)
            dataset_bootstrap_train.to_csv(ids_dir / ids_filename, index=False)

            ids_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}_test.csv'.format(i_bootstrap, i_n_subject_pairs)
            dataset_bootstrap_test.to_csv(ids_dir / ids_filename, index=False)


if __name__ == "__main__":
    main()
