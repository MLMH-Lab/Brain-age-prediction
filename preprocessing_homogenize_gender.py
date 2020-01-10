#!/usr/bin/env python3
"""Homogenize dataset.

We homogenize the dataset scanner_1 to not have a significant difference
between the proportion of men and women along the age. We used the
chi square test for homogeneity to verify if there is a difference.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from utils import load_demographic_data

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
                    help='Filename indicating the ids to be used.')
args = parser.parse_args()


def check_balance_across_groups(crosstab_df):
    """Verify if which age pair have gender imbalance."""
    combinations = list(itertools.combinations(crosstab_df.columns, 2))
    significance_level = 0.05 / len(combinations)

    for group1, group2 in combinations:
        contingency_table = crosstab_df[[group1, group2]]
        _, p_value, _, _ = stats.chi2_contingency(contingency_table, correction=False)

        if p_value < significance_level:
            return False, [group1, group2]

    return True, [None]


def get_problematic_group(crosstab_df):
    """Perform contingency analysis of the subjects gender."""
    balance_flag, problematic_groups = check_balance_across_groups(crosstab_df)

    if balance_flag:
        return None

    conditions_proportions = crosstab_df.apply(lambda r: r / r.sum(), axis=0)
    median_proportion = np.median(conditions_proportions.values[0, :])
    problematic_proportion = conditions_proportions[problematic_groups].values[0, :]

    problematic_group = problematic_groups[np.argmax(np.abs(problematic_proportion - median_proportion))]

    return problematic_group


def get_balanced_dataset(dataset_df):
    """Script to perform gender balancing across the subjects' age range."""

    while True:
        crosstab_df = pd.crosstab(dataset_df['Gender'], dataset_df['Age'])

        problematic_group = get_problematic_group(crosstab_df)

        if problematic_group is None:
            break

        condition_imbalanced = crosstab_df[problematic_group].idxmax()

        problematic_group_mask = (dataset_df['Age'] == problematic_group) & \
                                 (dataset_df['Gender'] == condition_imbalanced)

        list_to_drop = list(dataset_df[problematic_group_mask].sample(1).index)
        print('Dropping {:}'.format(dataset_df['Image_ID'].iloc[list_to_drop].values[0]))
        dataset_df = dataset_df.drop(list_to_drop, axis=0)

    return dataset_df


def main(experiment_name, scanner_name, input_ids_file):
    """Perform the exploratory data analysis."""
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids.csv'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    # Define random seed for sampling methods
    np.random.seed(42)

    dataset_df = load_demographic_data(participants_path, ids_path)

    dataset_balanced = get_balanced_dataset(dataset_df)

    homogeneous_ids_df = dataset_balanced[['Image_ID']]
    homogeneous_ids_df.to_csv(experiment_dir / 'homogenized_ids.csv', index=False)


if __name__ == "__main__":
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
