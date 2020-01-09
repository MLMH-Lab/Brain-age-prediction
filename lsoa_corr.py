#!/usr/bin/env python3
"""Script to assess correlations between age prediction error
and English indices of deprivation [1]

References:
[1] Department for Communities and Local Government. English Indices of Deprivation 2015. 2015.
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2015
"""

from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

PROJECT_ROOT = Path.cwd()


def ols_reg(df, x, y):
    """Perform linear regression using ordinary least squares (OLS) method"""

    y = np.asarray(df[y], dtype=float)
    x = np.asarray(sm.add_constant(df[x]), dtype=float)
    OLS_model = sm.OLS(y, x)
    OLS_results = OLS_model.fit()
    OLS_p = OLS_results.pvalues[1]
    OLS_coeff = OLS_results.params[1]

    return OLS_p, OLS_coeff


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    # ----------------------------------------------------------------------------------------
    correlation_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'correlation_analysis'

    ensemble_df = pd.read_csv(correlation_dir / 'ensemble_output.csv')
    variables_indices_df = pd.read_csv(correlation_dir / 'variables_indices_deprivation.csv')
    dataset = pd.merge(ensemble_df, variables_indices_df, on='Participant_ID')

    # Regression variables
    # Indices of deprivation variables exclude supplementary children and elderly measures
    y_list = ['BrainAGE_predmean', 'BrainAGER_predmean']
    x_list = ['IMD_decile', 'IMD_rank', 'IMD_score',
              'Income_deprivation_decile', 'Income_deprivation_rank', 'Income_deprivation_score',
              'Employment_deprivation_decile', 'Employment_deprivation_rank', 'Employment_deprivation_score',
              'Education_deprivation_decile', 'Education_deprivation_rank', 'Education_deprivation_score',
              'Health_deprivation_decile', 'Health_deprivation_rank', 'Health_deprivation_score',
              'Crime_decile', 'Crime_rank', 'Crime_score',
              'Barrier_housing_decile', 'Barrier_housing_rank', 'Barrier_housing_score',
              'Environment_deprivation_decile', 'Environment_deprivation_rank', 'Environment_deprivation_score']

    # Create empty dataframe for analysis of indices of deprivation
    indices_output = pd.DataFrame({'Row_labels_1': ['IMD_score', 'IMD_score',
                                                    'Income_deprivation_score', 'Income_deprivation_score',
                                                    'Employment_deprivation_score', 'Employment_deprivation_score',
                                                    'Education_deprivation_score', 'Education_deprivation_score',
                                                    'Health_deprivation_score', 'Health_deprivation_score',
                                                    'Crime_score', 'Crime_score',
                                                    'Barrier_housing_score', 'Barrier_housing_score',
                                                    'Environment_deprivation_score', 'Environment_deprivation_score'],
                                   'Row_labels_2': ['p_val', 'coef',
                                                    'p_val', 'coef',
                                                    'p_val', 'coef',
                                                    'p_val', 'coef',
                                                    'p_val', 'coef',
                                                    'p_val', 'coef',
                                                    'p_val', 'coef',
                                                    'p_val', 'coef']})
    indices_output.set_index('Row_labels_1', 'Row_labels_2')

    # create list of deprivation vars with scores only
    x_score = []
    for item in x_list:
        if item.split('_')[-1] == 'score':
            x_score.append(item)

    # Perform linear regression
    for y in y_list:
        y_results = []
        for x in x_score:
            dataset_var = dataset.dropna(subset=[x])
            OLS_p, OLS_coeff = ols_reg(dataset_var, y, x)
            y_results.append(OLS_p)
            y_results.append(OLS_coeff)
        indices_output[y] = y_results

    indices_output.to_csv(correlation_dir / 'indices_deprivation_ols_output.csv')


if __name__ == "__main__":
    main()
