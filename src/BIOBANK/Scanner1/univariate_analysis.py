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

import statsmodels.api as sm
import statsmodels.formula.api as smf


def check_missing(fs_df, dem_df):  # to do
    """Check if any participants in the FS dataset are not in the demographic dataset"""

    age_missing = []
    # for subject in fs_df['Participant_ID'].iteritems():
    #     if subject not in dem_df['Participant_ID'].iteritems():
    #         age_missing.append(subject)
    # print("Age missing for Participant_ID: ", age_missing)

    return age_missing


# test check_missing function
check_missing(dataset_fs_all_regions, dataset_demographic_excl_nan)


def normalise_region_df(df, region_name):
    """Normalise regional volume within df, add quadratic and cubic age vars"""

    normalised_df["Norm_vol_" + region_name] = df[region_name] / df['EstimatedTotalIntraCranialVol'] * 100

    return normalised_df


def main():  # to  do

    # Loading Freesurfer data
    dataset_fs_all_regions = pd.read_csv('/home/lea/PycharmProjects/'
                                         'predicted_brain_age/data/BIOBANK/Scanner1/freesurferData.csv')

    # Loading demographic data
    dataset_demographic = pd.read_csv('/home/lea/PycharmProjects/'
                                      'predicted_brain_age/data/BIOBANK/Scanner1/participants.tsv', sep='\t')
    dataset_demographic_excl_nan = dataset_demographic.dropna()

    # create a new col in FS dataset to contain Participant_ID
    dataset_fs_all_regions['Participant_ID'] = dataset_fs_all_regions['Image_ID']. \
        str.split('_', expand=True)[0]

    # check if any participants are missing demographic data
    # check_missing(dataset_fs_all_regions, dataset_demographic_excl_nan)

    # merge FS dataset and demographic dataset to access age
    dataset_fs_dem = pd.merge(dataset_fs_all_regions, dataset_demographic_excl_nan, on='Participant_ID')

    # create new df to add normalised regional volumes to
    normalised_df = pd.DataFrame(dataset_fs_dem[['Participant_ID', 'Diagn', 'Gender', 'Age']])
    normalised_df['Age2'] = normalised_df['Age'] * normalised_df['Age']
    normalised_df['Age3'] = normalised_df['Age'] * normalised_df['Age'] * normalised_df['Age']

    # to do: iterate over regions in df and run the below, changing 'region_name' in each iteration
    cols_to_ignore = ['Image_ID', 'Participant_ID', 'Dataset', 'Age', 'Gender', 'Diagn', 'EstimatedTotalIntraCranialVol']
    region_cols = []
    for col in dataset_fs_dem.columns:
        if col not in cols_to_ignore:
            region_cols.append(col)

    region_name = 'rh_supramarginal_volume'
    normalise_region_df(dataset_fs_dem, region_name)

    # output csv file with participant_id, age, age2, age3, normalised regional volume
    csv_normalised(normalised_array, region_name)

    # linear regression - ordinary least squares (OLS)
    endog = np.asarray(normalise_region_df['Norm_vol_' + region_name], dtype=float)
    exog = np.asarray(sm.add_constant(normalise_region_df[['Age', 'Age2', 'Age3']]), dtype=float)
    OLS_model = sm.OLS(endog, exog)
    OLS_results = OLS_model.fit()
    OLS_summary = OLS_results.summary()
    print(OLS_summary)
    OLS_coeff = OLS_results.params
    OLS_pvalue = OLS_results.pvalues
    OLS_conf = OLS_results.conf_int()
    OLS_result_df = pd.DataFrame({"pvalue":OLS_pvalue, "coeff":OLS_coeff})

    # example output to csv - different format required
    output_name = region_name + '_OLS_result.csv'
    output_path = '/home/lea/PycharmProjects/predicted_brain_age/outputs/' + output_name
    f = open(output_path, 'w')
    f.write(OLS_results.summary().as_csv())
    f.close()


if __name__ == "__main__":
    main()
