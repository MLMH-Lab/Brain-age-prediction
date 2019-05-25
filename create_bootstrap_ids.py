"""
Script to create gender-homogeneous bootstrap datasets to feed into create_h5_bootstrap script;
Creates 50 bootstrap samples with increasing size
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os

PROJECT_ROOT = Path.cwd()


def main():
    # Load final homogeneous dataset with Image IDs, age and gender variables
    dataset = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/homogeneous_dataset_freesurferData.h5', key='table')

    # Set random seed for random sampling of subjects
    np.random.seed(42)

    # Find range of ages in homogeneous dataset
    age_min = int(dataset['Age'].min()) # 47
    age_max = int(dataset['Age'].max()) # 73

    # Define or create directory to save bootstrap datasets
    bootstrap_dir = Path(str(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'bootstrap' / 'bootstrap_ids'))
    if not os.path.exists(str(bootstrap_dir)):
        os.makedirs(str(bootstrap_dir))

    # Loop to create 50 bootstrap samples that each contain 1 male, 1 female per age group/year
    for i in range(1,51):

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
        file_name = 'homogeneous_bootstrap_' + str(i) + '.csv'
        dataset_bootstrap.to_csv(str(bootstrap_dir / file_name), index=False, columns=['Image_ID'])


if __name__ == "__main__":
    main()