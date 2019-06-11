"""
Script to create gender-homogeneous bootstrap datasets to feed into create_h5_bootstrap script;
Creates 50 bootstrap samples with increasing size
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
    suffix_analysis_phase = '_homogeneous'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'dataset_homogeneous.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    bootstrap_dir = experiment_dir / 'bootstrap_analysis'
    bootstrap_dir.mkdir(exist_ok=True)
    ids_dir = bootstrap_dir / 'ids'
    ids_dir.mkdir(exist_ok=True)

    # Set random seed for random sampling of subjects
    np.random.seed(42)

    dataset = load_demographic_data(demographic_path, id_path)

    # Find range of ages in homogeneous dataset
    age_min = int(dataset['Age'].min())  # 47
    age_max = int(dataset['Age'].max())  # 73

    # Loop to create 50 bootstrap samples that each contain 1 male, 1 female per age group/year
    for i in range(1, 51):

        # Create empty df to add bootstrap subjects to
        dataset_bootstrap = pd.DataFrame(columns=dataset.columns)

        # Loop over ages
        for age in range(age_min, (age_max + 1)):

            # Get dataset for specific age only
            age_group = dataset.groupby('Age').get_group(age)

            # Loop over genders (0: female, 1:male)
            for gender in range(2):
                gender_group = age_group.groupby('Gender').get_group(gender)

                # Extract random subject of that gender and add to dataset_bootstrap
                random_row = gender_group.sample(n=i, replace=True)
                dataset_bootstrap = pd.concat([dataset_bootstrap, random_row])

        # Export dataset_bootstrap as csv
        ids_filename = 'homogeneous_bootstrap_{:02d}.csv'.format(i)
        dataset_bootstrap.to_csv(ids_dir / ids_filename, index=False, columns=['ID'])


if __name__ == "__main__":
    main()
