"""
Script to implement univariate analysis/logistic regression based on Zhao et al 2018, one per FS brain region
Step 1: normalise each brain region (create arrays of total brain region and specific brain region, then divide)
Step 2: create df with normalised brain region (dep var) and age of participant (indep var) (+ quadratic and cubic age?)
Step 3: output coefficient per subject


References
[1] - Zhao, Lu, et al. "Age-Related Differences in Brain Morphology and the Modifiers in Middle-Aged and Older Adults."
Cerebral Cortex (2018).
"""

import numpy as np
import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# from pathlib import Path

# load freesurfer dataset and change column names to a usable format
# PROJECT_ROOT = Path('../../../')
# dataset_fs_all_regions = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv')
dataset_fs_all_regions = pd.read_csv('/home/lea/PycharmProjects/'
                                     'predicted_brain_age/data/BIOBANK/Scanner1/freesurferData.csv')
dataset_fs_all_regions_renamed = dataset_fs_all_regions.rename(columns=lambda x: x.replace('-', '_'))

# load demographic dataset to access age of participants
# dataset_demographic = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv', sep='\t')
dataset_demographic = pd.read_csv('/home/lea/PycharmProjects/'
                                  'predicted_brain_age/data/BIOBANK/Scanner1/participants.tsv', sep='\t')
dataset_demographic_excl_nan = dataset_demographic.dropna()

# check if both datasets contain same number of participants
print(len(dataset_demographic_excl_nan))
print(len(dataset_fs_all_regions_renamed))


def normalise_region(total_volume, region_volume):
    """Normalise regional volume, extract participant IDs,
    and return array of participant IDs and normalised regional volumes"""

    total = np.array(dataset_fs_all_regions_renamed[total_volume])
    region = np.array(dataset_fs_all_regions_renamed[region_volume])
    region_normalised = region / total

    # Create new column Participant_ID from Image_ID
    dataset_fs_all_regions_renamed["Participant_ID"] = dataset_fs_all_regions_renamed['Image_ID']. \
        str.split('_', expand=True)[0]
    participant_id = np.array(dataset_fs_all_regions_renamed["Participant_ID"])

    # Create new array
    normalised_array = np.array([participant_id, region_normalised])
    return normalised_array


# test function
normalise_region('EstimatedTotalIntraCranialVol', 'Left_Lateral_Ventricle')


def main():
    pass


if __name__ == "__main__":
    main()
