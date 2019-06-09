"""
Script to assess sample homogeneity: gender and ethnicity


"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def check_balance_across_groups(crosstab_df):
    """"""
    combinations = list(itertools.combinations(crosstab_df.columns, 2))
    significance_level = 0.05 / len(combinations)

    for group1, group2 in combinations:
        # print('{} vs {}'.format(group1, group2))
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
    problematic = conditions_proportions[problematic_groups].values[0, :]

    problematic_group = problematic_groups[np.argmax(np.abs(problematic - median_proportion))]

    return problematic_group


def get_balanced_dataset(dataset):
    """"""

    while True:
        crosstab_df = pd.crosstab(dataset['Gender'], dataset['Age'])

        problematic_group = get_problematic_group(crosstab_df)

        if problematic_group is None:
            break

        condition_unbalanced = crosstab_df[problematic_group].idxmax()

        problematic_group_mask = (dataset['Age'] == problematic_group) & \
                                 (dataset['Gender'] == condition_unbalanced)

        list_to_drop = list(dataset[problematic_group_mask].sample(1).index)
        print('Dropping {:}'.format(dataset['ID'].iloc[list_to_drop].values[0]))
        dataset = dataset.drop(list_to_drop, axis=0)

    return dataset


def main():
    """Perform the exploratory data analysis."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    suffix_analysis_phase = '_homogeneous'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    # Define random seed for sampling methods
    np.random.seed(42)

    dataset = load_demographic_data(demographic_path, id_path)

    dataset_balanced = get_balanced_dataset(dataset)

    homogeneous_ids = pd.DataFrame(dataset_balanced['ID'])
    homogeneous_ids.to_csv(experiment_dir / ('dataset' + suffix_analysis_phase + '.csv'), index=False)


if __name__ == "__main__":
    main()
