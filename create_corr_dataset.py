"""Script to create dataset for var_corr and lsoa_corr
Step 1: Create variables for average age predictions and prediction errors, incl. BrainAGE, BrainAGER [1]
Step 2: Add demographic variables from UK Biobank
Step 3: Add demographic variables from English Indices of Deprivation [2]

References:
[1] Le TT, Kuplicki RT, McKinney BA, Yeh H-W, Thompson WK, Paulus MP and Tulsa 1000 Investigators (2018)
A Nonlinear Simulation Framework Supports Adjusting for Age When Analyzing BrainAGE.
Front. Aging Neurosci. 10:317. doi: 10.3389/fnagi.2018.00317
[2] Eepartment for Communities and Local Government. English indicesof deprivation 2015. 2015.
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2015
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():

    # Extract participant ID in age_pred to match ID format in demographic dataset
    age_pred['ID'] = age_pred['Participant_ID'].str.split('-', expand=True)[1]
    age_pred['ID'] = pd.to_numeric(age_pred['ID'])

    # Loading demographic data in UK Biobank to access variables
    dataset_dem = pd.read_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'),
                              usecols=['eid',
                                       '6138-2.0', '6138-2.1', '6138-2.2', '6138-2.3', '6138-2.4',
                                       '22702-0.0', '22704-0.0',
                                       '24005-0.0',
                                       '24009-0.0', '24010-0.0', '24014-0.0',
                                       '24500-0.0', '24501-0.0', '24502-0.0', '24506-0.0'])
    dataset_dem.columns = ['ID',
                           'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
                           'East_coordinate', 'North_coordinate',
                           'Air_pollution',
                           'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
                           'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Create new education cols to simulate ordinal scale
    # The original 6-point education scale was reduced to a 4-point scale using the following assumptions:
    # Codes 3 "O levels / GCSEs or equivalent" and 4 "CSEs or equivalent" are equivalent
    # Codes 5 "NVQ or HND or HNC or equivalent" and 6 "Other professional qualifications" are equivalent
    # Codes -7	"None of the above" and -3	"Prefer not to answer" are treated as missing data
    education_dict = {1: 4, 2: 2, 3: 1, 4: 1, 5: 3, 6: 3}
    dataset_dem['Education_1'] = dataset_dem['Education_1'].map(education_dict)
    dataset_dem['Education_2'] = dataset_dem['Education_2'].map(education_dict)
    dataset_dem['Education_3'] = dataset_dem['Education_3'].map(education_dict)
    dataset_dem['Education_4'] = dataset_dem['Education_4'].map(education_dict)
    dataset_dem['Education_5'] = dataset_dem['Education_5'].map(education_dict)

    # Create col for maximum of education level per respondent
    dataset_dem['Education_highest'] = dataset_dem[['Education_1', 'Education_2', 'Education_3',
                                                    'Education_4', 'Education_5']].apply(max, axis=1)

    # Merge age_pred and dataset_dem datasets
    dataset = pd.merge(age_pred, dataset_dem, on='ID')

    # Loading data from English Indices of Deprivation 2015
    dataset_imd = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'IMD_data.csv',
                              usecols=['Participan', 'code', 'name', 'label',
                                       'wide index of multiple deprivation LSOA_IMD_decile',
                                       'wide index of multiple deprivation LSOA_IMD_rank',
                                       'wide index of multiple deprivation LSOA_IMD_score',
                                       'wide index of multiple deprivation LSOA_income_dep_domain_decile',
                                       'wide index of multiple deprivation LSOA_income_dep_domain_rank',
                                       'wide index of multiple deprivation LSOA_income_dep_domain_score',
                                       'wide index of multiple deprivation LSOA_employ_dep_decile',
                                       'wide index of multiple deprivation LSOA_employ_dep_rank',
                                       'wide index of multiple deprivation LSOA_employ_dep_score',
                                       'wide index of multiple deprivation LSOA_EST_decile',
                                       'wide index of multiple deprivation LSOA_EST_rank',
                                       'wide index of multiple deprivation LSOA_EST_score',
                                       'wide index of multiple deprivation LSOA_HDD_decile',
                                       'wide index of multiple deprivation LSOA_HDD_rank',
                                       'wide index of multiple deprivation LSOA_HDD_score',
                                       'wide index of multiple deprivation LSOA_crime_decile',
                                       'wide index of multiple deprivation LSOA_crime_rank',
                                       'wide index of multiple deprivation LSOA_crime_score',
                                       'wide index of multiple deprivation LSOA_BHS_decile',
                                       'wide index of multiple deprivation LSOA_BHS_rank',
                                       'wide index of multiple deprivation LSOA_BHS_score',
                                       'wide index of multiple deprivation LSOA_LED_decile',
                                       'wide index of multiple deprivation LSOA_LED_rank',
                                       'wide index of multiple deprivation LSOA_LED_score',
                                       'wide index of multiple deprivation LSOA_IDchild_decile',
                                       'wide index of multiple deprivation LSOA_IDchild_rank',
                                       'wide index of multiple deprivation LSOA_IDchild_score',
                                       'wide index of multiple deprivation LSOA_IDelder_decile',
                                       'wide index of multiple deprivation LSOA_IDelder_rank',
                                       'wide index of multiple deprivation LSOA_IDelder_score'])
    dataset_imd.columns = ['Participant_ID', 'LSOA_code', 'LSOA_name', 'LSOA_label',
                           'IMD_decile', 'IMD_rank', 'IMD_score',
                           'Income_deprivation_decile', 'Income_deprivation_rank', 'Income_deprivation_score',
                           'Employment_deprivation_decile', 'Employment_deprivation_rank',
                           'Employment_deprivation_score',
                           'Education_deprivation_decile', 'Education_deprivation_rank', 'Education_deprivation_score',
                           'Health_deprivation_decile', 'Health_deprivation_rank', 'Health_deprivation_score',
                           'Crime_decile', 'Crime_rank', 'Crime_score',
                           'Barrier_housing_decile', 'Barrier_housing_rank', 'Barrier_housing_score',
                           'Environment_deprivation_decile', 'Environment_deprivation_rank',
                           'Environment_deprivation_score',
                           'Income_deprivation_aff_children_decile', 'Income_deprivation_aff_children_rank',
                           'Income_deprivation_aff_children_score',
                           'Income_deprivation_aff_elder_decile', 'Income_deprivation_aff_elder_rank',
                           'Income_deprivation_aff_elder_score']

    # Merge dataset with dataset_imd
    dataset = pd.merge(dataset, dataset_imd, on='Participant_ID')

    # output csv with age variables and demographic variables
    dataset.to_csv(str(output_dir / 'age_predictions_demographics.csv'))


if __name__ == "__main__":
    main()
