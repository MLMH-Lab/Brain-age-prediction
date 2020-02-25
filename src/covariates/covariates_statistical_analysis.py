#!/usr/bin/env python3
"""Script to assess correlations between BrainAGE/BrainAGER and demographic variables in UK Biobank
(dataset created in variables_biobank.py)"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

parser.add_argument('-M', '--model_name',
                    dest='model_name',
                    help='Name of the model.')

args = parser.parse_args()


def cohend(d1, d2):
    """Calculate Cohen's d effect size for independent samples"""

    n1, n2 = len(d1), len(d2)
    d1_variance, d2_variance = np.var(d1, ddof=1), np.var(d2, ddof=1)
    std_pooled = np.sqrt(((n1 - 1) * d1_variance + (n2 - 1) * d2_variance) / (n1 + n2 - 2))
    d1_mean, d2_mean = np.mean(d1), np.mean(d2)
    effect_size = (d1_mean - d2_mean) / std_pooled

    return effect_size


def main(experiment_name, scanner_name, model_name):
    """"""
    correlation_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'correlation_analysis'
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'

    ensemble_df = pd.read_csv(correlation_dir / f'ensemble_{model_name}_output.csv')
    ensemble_df['id'] = ensemble_df['image_id'].str.split('_').str[0]
    ensemble_df['id'] = ensemble_df['id'].str.split('-').str[1]
    ensemble_df['id'] = pd.to_numeric(ensemble_df['id'])

    participants = pd.read_csv(participants_path, sep='\t')

    ensemble_df['participant_id'] = ensemble_df['image_id'].str.split('_').str[0]

    dataset = pd.merge(ensemble_df, participants[['participant_id', 'Gender']], on='participant_id')

    # Correlation variables
    y_list = ['BrainAGE_predmean', 'BrainAGER_predmean']

    dataset_f = dataset[dataset['Gender'] == 0]
    dataset_m = dataset[dataset['Gender'] == 1]

    # Create empty dataframe for analysis of education level
    output = pd.DataFrame({'Row_labels_1': ['female vs male',
                                                      'female vs male',
                                                      ],
                                     'Row_labels_2': ['p_val', 'cohen']})
    output.set_index('Row_labels_1', 'Row_labels_2')

    # Independent t-tests with alpha corrected for multiple comparisons using Bonferroni's method
    alpha_corrected = 0.05 / 2

    for y in y_list:
        y_results = []

        print('\n', y)
        tstat, pval = ttest_ind(dataset_f[y], dataset_m[y])
        effect_size = cohend(dataset_f[y], dataset_m[y])
        y_results.append(pval)
        y_results.append(effect_size)
        if pval < alpha_corrected:
            print(f"female vs male [t-test pval, cohen's d]: {pval:6.3}, {effect_size:6.3}")

        output[y] = y_results

    output.to_csv(correlation_dir / f'gender_ttest_{model_name}output.csv')

if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.model_name)
