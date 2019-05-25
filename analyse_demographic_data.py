"""Script to assess sample homogeneity in UK BIOBANK Scanner1: gender and ethnicity

Due to the limited number of non-white ethnicity we decided to exclude them from further analysis.

Step 1: Organising dataset
Step 2: Visualisation of distribution
Step 3: Chi-square contingency analysis
Step 4: Remove subjects based on chi-square results to achieve homogeneous sample in terms of gender and ethnicity

Supplementary data and labels acquired from https://biobank.ctsu.ox.ac.uk/crystal/search.cgi
"""
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()


def save_frequency_table(input_df, col_name):
    """Export frequency table of column as csv"""

    freq_table = input_df[col_name].value_counts()
    print(col_name)
    print(freq_table)
    file_name = col_name + '_freq_table.csv'
    freq_table.to_csv(PROJECT_ROOT / 'outputs' / file_name)


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


def get_balanced_dataset(demographic_df):
    """"""
    while True:
        crosstab_df = pd.crosstab(demographic_df['Gender'], demographic_df['Age'])

        problematic_group = get_problematic_group(crosstab_df)

        if problematic_group is None:
            break

        condition_unbalanced = crosstab_df[problematic_group].idxmax()

        problematic_group_mask = (demographic_df['Age'] == problematic_group) & \
                                 (demographic_df['Gender'] == condition_unbalanced)

        list_to_drop = list(demographic_df[problematic_group_mask].sample(1).index)
        print('Dropping {:}'.format(list_to_drop))
        demographic_df = demographic_df.drop(list_to_drop, axis=0)

    return demographic_df


def plot_save_histogram(male_ages, female_ages):
    """Create histogram of age distribution by gender and saves in output folder as png"""
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 7))

    plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='male')
    plt.hist(female_ages, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='female')

    plt.title("Age distribution in UK BIOBANK by gender", fontsize=17)
    plt.axis('tight')
    plt.xlabel("Age at MRI scan [years]", fontsize=15)
    plt.ylabel("Number of subjects", fontsize=15)
    plt.xticks(range(45, 75, 5))
    plt.yticks(range(0, 401, 50))
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)

    plt.savefig(str(PROJECT_ROOT / 'outputs' / 'gender_age_dist_BIOBANK.png'))


def main():
    # Define random seed for sampling methods
    np.random.seed(42)

    # Load freesurfer data
    dataset_fs = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv')

    # Loading supplementary demographic data
    dataset_dem = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv',
                              usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    dataset_dem.columns = ['ID', 'Gender', 'Ethnicity', 'Age']
    dataset_dem = dataset_dem.dropna()

    # Create a new 'ID' column in FS dataset to match supplementary demographic data
    dataset_fs['ID'] = dataset_fs['Image_ID'].str.split('_', expand=True)[0]
    dataset_fs['ID'] = dataset_fs['ID'].str.split('-', expand=True)[1]
    dataset_fs['ID'] = pd.to_numeric(dataset_fs['ID'])

    # Merge supplementary demographic data with freesurfer data
    dataset = pd.merge(dataset_fs, dataset_dem, on='ID')

    # Labeling data
    ethnicity_dict = {
        1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
        2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
        3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
        4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
        5: 'Chinese',
        6: 'Other',
        -1: 'Not known', -3: 'Not known'
    }

    gender_dict = {
        0: 'Female',
        1: 'Male'
    }

    dataset = dataset.replace({'Gender': gender_dict})
    dataset = dataset.replace({'Ethnicity': ethnicity_dict})

    # Export ethnicity and age distribution for future reference
    save_frequency_table(dataset, 'Ethnicity')
    save_frequency_table(dataset, 'Age')

    # Exclude ages with <100 participants,
    dataset = dataset[dataset['Age'] > 46]

    # Exclude non-white ethnicities due to small subgroups
    dataset = dataset[dataset['Ethnicity'] == 'White']

    dataset = get_balanced_dataset(dataset)

    chi2_stats, p_value, _, _ = stats.chi2_contingency(pd.crosstab(dataset['Gender'], dataset['Age']))
    print('chi2 statistics {:4.2f} p value {:4.3}'.format(chi2_stats, p_value))

    homogeneous_ids = pd.DataFrame(dataset['Image_ID'])
    homogeneous_ids.to_csv(PROJECT_ROOT / 'outputs' / 'homogeneous_dataset.csv', index=False)

    male_ages = dataset.groupby('Gender').get_group('Male').Age
    female_ages = dataset.groupby('Gender').get_group('Female').Age

    plot_save_histogram(male_ages, female_ages)

    print('Whole dataset')
    print(dataset.Age.describe())
    dataset.Age.describe().to_csv(PROJECT_ROOT / 'outputs' / 'scanner01_whole_dataset_dem.csv')

    print('Grouped dataset')
    print(dataset.groupby('Gender').Age.describe())
    dataset.groupby('Gender').Age.describe().to_csv(PROJECT_ROOT / 'outputs' / 'scanner01_grouped_dem.csv')


if __name__ == "__main__":
    main()
