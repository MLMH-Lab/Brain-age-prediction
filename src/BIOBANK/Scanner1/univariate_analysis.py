"""
Script to implement univariate analysis/linear mixed-effects regression based on [1], one per FS brain region
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

# Load freesurfer dataset
PROJECT_ROOT = Path('../../../../')

dataset_fs_all_regions = pd.read_csv('/home/lea/PycharmProjects/'
                                     'predicted_brain_age/data/BIOBANK/Scanner1/freesurferData.csv')


# load demographic dataset to access age of participants
dataset_demographic = pd.read_csv('/home/lea/PycharmProjects/'
                                  'predicted_brain_age/data/BIOBANK/Scanner1/participants.tsv', sep='\t')
dataset_demographic_excl_nan = dataset_demographic.dropna()

# check if both datasets contain same number of participants
print(len(dataset_demographic_excl_nan))
print(len(dataset_fs_all_regions))


def extract_df(region_volume):
    """Create a new dataset that only contains columns relevant to univariate analysis,
    add participant_id and demographics,
    check if/which subjects with FS data are missing demographic data"""

    dataset_region = dataset_fs_all_regions[['Image_ID', 'EstimatedTotalIntraCranialVol', region_volume]].copy()

    # extract Participant_ID from Image_ID
    dataset_region['Participant_ID'] = dataset_region['Image_ID']. \
        str.split('_', expand=True)[0]

    # create list with subjects in FS output with missing age data
    age_missing = []
    for subject in dataset_region['Participant_ID'].iteritems():
        if subject not in dataset_demographic_excl_nan['Participant_ID'].iteritems():
            age_missing += subject
    print("Age missing for Participant_ID: ", age_missing)

    # merge FS dataset with demographics dataset to access age, gender, diagnosis
    global dataset_region_age
    dataset_region_age = pd.merge(dataset_region, dataset_demographic_excl_nan, on='Participant_ID')

    return dataset_region_age


# test extract_df function
extract_df('Left-Lateral-Ventricle')


# attempt to normalise within df to preserve var labels - TO DO
# def normalise_region_df(region_volume):
#     """Normalise regional volume using df"""
#
#     new_norm_df = dataset_region_age['EstimatedTotalIntraCranialVol'].divide(dataset_region_age[region_volume])
#
#     return new_norm_df
#
# # test normalise_region function
# normalise_region_df('Left-Lateral-Ventricle')


def normalise_region(region_volume):
    """Normalise regional volume"""

    total = np.array(dataset_region_age['EstimatedTotalIntraCranialVol'])
    region = np.array(dataset_region_age[region_volume])
    region_normalised = region / total

    # Create new array with relevant variables
    participant_id = np.array(dataset_region_age['Participant_ID'])
    age = np.array(dataset_region_age['Age'])
    age2 = age * age
    age3 = age * age * age

    global normalised_array
    normalised_array = np.array([participant_id, age, age2, age3, region_normalised])

    return normalised_array


# test normalise_region function
normalise_region('Left-Lateral-Ventricle')


# def csv_normalised(normalised_array, region_name):
#     """Write normalised array to csv file by converting into df"""
#
#     file_name = region_name + '_normalised_array.csv'
#     pd.DataFrame(normalised_array).to_csv('/Users/leabaecker/PycharmProjects/predicted_brain_age/outputs/'
#                                           + file_name,
#                                           columns=[{'Participant_ID':normalised_array[:,0], 'Age':normalised_array[:,1],
#                                                     'Age2':normalised_array[:,2], 'Age3':normalised_array[:,3],
#                                                     'Normalised_vol':normalised_array[:,4]}])
#
#
# # test csv_normalised function
# csv_normalised(normalised_array, 'Left-Lateral-Ventricle')

def main(): # to  do
    # Loading Freesurfer data
    dataset_fs_all_regions = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv')

    # Loading demographic data
    dataset_demographic = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv', sep='\t')

    to

    extract_df(region_volume)
    normalise_region(region_volume)

    pass


if __name__ == "__main__":
    main()
