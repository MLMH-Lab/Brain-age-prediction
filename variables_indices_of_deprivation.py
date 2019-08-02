"""Create dataset with variables from English Indices of Deprivation, publicly available online [1];
This dataset was merged with the UK Biobank data based on the participants' home coordinates using the QGIS application

References:
[1] Department for Communities and Local Government. English Indices of Deprivation 2015. 2015.
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2015
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    correlation_dir = experiment_dir / 'correlation_analysis'
    correlation_dir.mkdir(exist_ok=True)

    # Loading data from English Indices of Deprivation 2015
    indices_deprivation_df = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'IMD_data.csv',
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
    indices_deprivation_df.columns = ['Participant_ID', 'LSOA_code', 'LSOA_name', 'LSOA_label',
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

    # output csv with age variables and demographic variables
    indices_deprivation_df.to_csv(correlation_dir / 'variables_indices_deprivation.csv', index=False)

if __name__ == "__main__":
    main()