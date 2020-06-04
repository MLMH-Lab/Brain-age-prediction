#!/usr/bin/env python3
"""
Script to implement univariate analysis based on [1], regression for age and volume per region
Step 1: normalise each brain region
Step 2: create df with normalised brain region (dep var) and age of participant (indep var) (+ quadratic and cubic age)
Step 3: output coefficient per subject

References
[1] - Zhao, Lu, et al. "Age-Related Differences in Brain Morphology and the Modifiers in Middle-Aged and Older Adults."
Cerebral Cortex (2018).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import COLUMNS_NAME, load_freesurfer_dataset

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='homogenized_ids.csv',
                    help='Filename indicating the ids to be used.')

args = parser.parse_args()


def normalise_region_df(df, region_name):
    """Normalise region by total intracranial volume"""
    return df[region_name] / df['EstimatedTotalIntraCranialVol'] * 100


def linear_regression(df, region_name):
    """Perform linear regression using ordinary least squares (OLS) method"""

    endog = df['Norm_vol_' + region_name].values
    exog = sm.add_constant(df[['Age', 'Age^2', 'Age^3']].values)

    OLS_model = sm.OLS(endog, exog)

    OLS_results = OLS_model.fit()

    return OLS_results.params, OLS_results.bse, OLS_results.tvalues, OLS_results.pvalues


def main(experiment_name, scanner_name, input_ids_file):
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'freesurferData.csv'
    ids_path = experiment_dir / input_ids_file

    # Create experiment's output directory
    univariate_dir = experiment_dir / 'univariate_analysis'
    univariate_dir.mkdir(exist_ok=True)

    dataset = load_freesurfer_dataset(participants_path, ids_path, freesurfer_path)

    # Create new df to add normalised regional volumes to
    normalised_df = dataset[['participant_id', 'Diagn', 'Gender', 'Age']]
    normalised_df['Age^2'] = normalised_df['Age'] ** 2
    normalised_df['Age^3'] = normalised_df['Age'] ** 3

    # Create empty df for regression output; regions to be added
    regression_output = pd.DataFrame({'Row_labels_stat': ['Coeff', 'Coeff', 'Coeff', 'Coeff',
                                                          'std_err', 'std_err', 'std_err', 'std_err',
                                                          't_stats', 't_stats', 't_stats', 't_stats',
                                                          'p_val', 'p_val', 'p_val', 'p_val'],

                                      'Row_labels_exog': ['Constant', 'Age', 'Age2', 'Age3',
                                                          'Constant', 'Age', 'Age2', 'Age3',
                                                          'Constant', 'Age', 'Age2', 'Age3',
                                                          'Constant', 'Age', 'Age2', 'Age3']})

    regression_output.set_index('Row_labels_stat', 'Row_labels_exog')

    for region_name in COLUMNS_NAME:
        print(region_name)
        normalised_df['Norm_vol_' + region_name] = normalise_region_df(dataset, region_name)

        # Linear regression - ordinary least squares (OLS)
        coeff, std_err, t_value, p_value = linear_regression(normalised_df, region_name)

        regression_output[region_name] = np.concatenate((coeff, std_err, t_value, p_value), axis=0)

    # Output to csv
    regression_output.to_csv(univariate_dir / 'OLS_result.csv', index=False)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
