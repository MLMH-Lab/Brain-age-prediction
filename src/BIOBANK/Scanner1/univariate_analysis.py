"""
Script to implement univariate analysis/logistic regression based on Zhao et al 2018, one per FS brain region
Step 1: normalise each brain region (create arrays of total brain region and specific brain region, then divide)
Step 2: create df with normalised brain region (dep var) and age of participant (indep var) (+ quadratic and cubic age?)
Step 3: output coefficient per subject


References
[1] - Zhao, Lu, et al. "Age-Related Differences in Brain Morphology and the Modifiers in Middle-Aged and Older Adults."
Cerebral Cortex (2018).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path('../../../')  # does not work for loading the below file
dataset_fs_all_regions = pd.read_csv('/home/lea/PycharmProjects/'
                                     'predicted_brain_age/data/BIOBANK/Scanner1/freesurferData.csv')

# some column names contain '-' which is not allowed, so they are replaced with '_'
dataset_fs_all_regions_renamed = dataset_fs_all_regions.rename(columns=lambda x: x.replace('-', '_'))
dataset_fs_all_regions_renamed.columns


def normalise_region(total_volume, region_volume):
    """Normalise each brain region by dividing the regional by the total intracranvial volume"""
    total = np.array(dataset_fs_all_regions_renamed[total_volume])
    region = np.array(dataset_fs_all_regions_renamed[region_volume])
    region_normalised = region / total
    print(region_normalised)


# test function
normalise_region('EstimatedTotalIntraCranialVol', 'Left_Lateral_Ventricle')


def main():
    pass


if __name__ == "__main__":
    main()
