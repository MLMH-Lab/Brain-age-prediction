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


def normalise_region_df(df, region_name):
    """Normalise regional volume within df, add quadratic and cubic age vars"""

    normalised_df["Norm_vol_" + region_name] = df[region_name] / df['EstimatedTotalIntraCranialVol'] * 100

    return normalised_df


def ols_reg(df, region_name):
    """Perform linear regression using ordinary least squares (OLS) method"""

    endog = np.asarray(df['Norm_vol_' + region_name], dtype=float)
    exog = np.asarray(sm.add_constant(df[['Age', 'Age2', 'Age3']]), dtype=float)
    OLS_model = sm.OLS(endog, exog)
    OLS_results = OLS_model.fit()

    # access regression results
    OLS_coeff = pd.DataFrame(OLS_results.params)
    OLS_pvalue = pd.DataFrame(OLS_results.pvalues)
    OLS_tvalue = pd.DataFrame(OLS_results.tvalues)
    OLS_se = pd.DataFrame(OLS_results.bse)

    # add to reg_output df
    OLS_df = pd.concat([OLS_coeff, OLS_se, OLS_tvalue, OLS_pvalue], ignore_index=True)
    reg_output[region_name] = OLS_df
    # reg_output.reset_index(drop=True)

    return reg_output


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

    # merge FS dataset and demographic dataset to access age
    dataset_fs_dem = pd.merge(dataset_fs_all_regions, dataset_demographic_excl_nan, on='Participant_ID')

    # create new df to add normalised regional volumes to
    global normalised_df
    normalised_df = pd.DataFrame(dataset_fs_dem[['Participant_ID', 'Diagn', 'Gender', 'Age']])
    normalised_df['Age2'] = normalised_df['Age'] * normalised_df['Age']
    normalised_df['Age3'] = normalised_df['Age'] * normalised_df['Age'] * normalised_df['Age']

    # create empty df for regression output; regions to be added
    global reg_output
    reg_output = pd.DataFrame({"Row_labels_stat": ['Coeff', 'Coeff', 'Coeff', 'Coeff',
                                                   'std_err', 'std_err', 'std_err', 'std_err',
                                                   't', 't', 't', 't',
                                                   'p_val', 'p_val', 'p_val', 'p_val'],
                               "Row_labels_exog": ['Constant', 'Age', 'Age2', 'Age3',
                                                   'Constant', 'Age', 'Age2', 'Age3',
                                                   'Constant', 'Age', 'Age2', 'Age3',
                                                   'Constant', 'Age', 'Age2', 'Age3']})
    reg_output.set_index('Row_labels_stat', 'Row_labels_exog')

    # update normalised_df to contain normalised regions for all regions
    cols_to_ignore = ['Image_ID', 'Participant_ID', 'Dataset', 'Age', 'Gender', 'Diagn', 'EstimatedTotalIntraCranialVol']
    region_cols = []
    for col in dataset_fs_dem.columns:
        if col not in cols_to_ignore:
            region_cols.append(col)

    for region in region_cols:
        normalise_region_df(dataset_fs_dem, region)

        # linear regression - ordinary least squares (OLS)
        ols_reg(normalised_df, region)

    # output to csv
    reg_output.to_csv('/home/lea/PycharmProjects/predicted_brain_age/outputs/OLS_result.csv', index=False)


if __name__ == "__main__":
    main()
