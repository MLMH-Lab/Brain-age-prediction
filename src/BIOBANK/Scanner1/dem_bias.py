"""Script to assess sample homogeneity in UK BIOBANK Scanner1: gender vs age, ethnicity vs age;
Supplementary data and labels acquired from https://biobank.ctsu.ox.ac.uk/crystal/search.cgi

Step 1: Organising dataset
Step 2: Visualisation of distribution
Step 3: Chi-square contingency analysis
Step 4: Remove subjects based on chi-square results to achieve homogeneous sample in terms of gender and ethnicity"""
import itertools

import pandas as pd
import numpy as np
import scipy.stats as stats


def save_fre_table(input_df, col_name):
    """Export frequency table of column as csv"""

    fre_table = input_df[col_name].value_counts()
    file_name = col_name + '_fre_table.csv'
    fre_table.to_csv('/home/lea/PycharmProjects/predicted_brain_age/outputs/' + file_name)


def chi2_contingency_test(crosstab_df, age_combinations, sig_list, age1, age2):
    """Perform multiple 2x2 Pearson chi-square analyses, corrected for multiple comparisons"""

    contingency_table = crosstab_df[[age1, age2]]
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table, correction=False)

    # Bonferroni correction for multiple comparisons; use sig_list to check which ages are most different
    sig_level = 0.05 / len(age_combinations)
    msg = "Chi-square test for ages {} vs {} is significant:\nTest Statistic: {}\np-value: {}\n"
    if p_value < sig_level:
        sig_list.append(age1)
        sig_list.append(age2)
        # print(msg.format(age1, age2, chi2, p_value))


def chi2_contingency_analysis(demographics_data_df):
    """Perform contingency analysis of the subjects gender."""

    # Create list of unique age combinations for chi2 contingency analysis
    gender_observed = pd.crosstab(demographics_data_df['Gender'], demographics_data_df['Age'])
    age_list = list(gender_observed.columns)
    age_combinations = list(itertools.product(age_list, age_list))
    age_combinations_new = []
    for age_tuple in age_combinations:
        if (age_tuple[1], age_tuple[0]) not in age_combinations_new:
            if age_tuple[0] != age_tuple[1]:
                age_combinations_new.append(age_tuple)

    # Perform chi2 contingency analysis for each age combination
    # sig_list is used to keep track of ages where gender proportion is significantly different
    sig_list = []
    for age_tuple in age_combinations_new:
        chi2_contingency_test(gender_observed, age_combinations_new, sig_list, age_tuple[0], age_tuple[1])

    # Assess how often each age is significantly different from the others
    dict_sig = {}
    for item in sig_list:
        if item in dict_sig:
            dict_sig[item] += 1
        elif item not in dict_sig:
            dict_sig[item] = 1
        else:
            print("error with " + str(item))

    return dict_sig, gender_observed


def get_ids_to_drop(input_df, age, gender, n_to_drop):
    """Extract random sample of participant IDs per age per gender to drop from total sample"""

    df_filtered = input_df[(input_df['Age'] == age) & (input_df['Gender'] == gender)]

    # random sample of IDs to drop
    df_to_drop = df_filtered.sample(n_to_drop)
    id_list = list(df_to_drop['ID'])

    return id_list

def balancing_sample(demographics_data_df, dict_sig, gender_observed):
    """Fix gender balance."""

    print('Fixing unbalance...')
    # Undersample the more prominent gender per age in dict_sig and store removed IDs in ids_to_drop
    ids_to_drop = []

    for key in dict_sig.keys():

        if dict_sig[key] > 4:
            if gender_observed.loc['Female', key] > gender_observed.loc['Male', key]:
                gender_higher = 'Female'
            elif gender_observed.loc['Female', key] < gender_observed.loc['Male', key]:
                gender_higher = 'Male'
            else:
                print("Error with: " + str(key))

            gender_diff = gender_observed.loc['Female', key] - gender_observed.loc['Male', key]
            diff_to_remove = int(abs(gender_diff) * 0.5)

            ids_list = get_ids_to_drop(demographics_data_df, key, gender_higher, diff_to_remove)
            ids_to_drop.extend(ids_list)

    return demographics_data_df[~demographics_data_df.ID.isin(ids_to_drop)]


def main():
    # Define random seed for sampling methods
    np.random.seed = 123

    # Load freesurfer data
    dataset_fs = pd.read_csv(
        '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/freesurferData.csv')

    # Create a new 'eid' col in FS dataset to match supplementary demographic data
    dataset_fs['Participant_ID'] = dataset_fs['Image_ID']. \
        str.split('_', expand=True)[0]
    dataset_fs['ID'] = dataset_fs['Participant_ID']. \
        str.split('-', expand=True)[1]

    # Loading supplementary demographic data
    dataset_dem = pd.read_csv(
        '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/ukb22321.csv',
        usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    dataset_dem.columns = ['ID', 'Gender', 'Ethnicity', 'Age']
    dataset_dem_excl_nan = dataset_dem.dropna()

    # Labeling data
    grouped_ethnicity_dict = {
        1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
        2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
        3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
        4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
        5: 'Chinese',
        6: 'Other',
        -1: 'Not known', -3: 'Not known'
    }

    gender_dict = {
        0: 'Female', 1: 'Male'
    }

    dataset_dem_excl_nan = dataset_dem_excl_nan.replace({'Gender': gender_dict})
    dataset_dem_excl_nan_grouped = dataset_dem_excl_nan.replace({'Ethnicity': grouped_ethnicity_dict})

    # Export ethnicity and age distribution for future reference
    save_fre_table(dataset_dem_excl_nan_grouped, 'Ethnicity')
    save_fre_table(dataset_dem_excl_nan_grouped, 'Age')

    # Exclude ages with <100 participants, exclude non-white ethnicities due to small subgroups
    dataset_dem_ab46 = dataset_dem_excl_nan_grouped[dataset_dem_excl_nan_grouped['Age'] > 46]
    dataset_dem_ab46_ethn = dataset_dem_ab46[dataset_dem_ab46['Ethnicity'] == 'White']

    dict_sig, gender_observed = chi2_contingency_analysis(dataset_dem_ab46_ethn)
    print('Unbalanced groups')
    print(dict_sig)

    # Create new balanced dataset
    reduced_dataset = balancing_sample(dataset_dem_ab46_ethn, dict_sig, gender_observed)

    dict_sig, gender_observed = chi2_contingency_analysis(reduced_dataset)
    print('Unbalanced groups')
    print(dict_sig)

    # Output final dataset
    reduced_dataset.to_csv('/home/lea/PycharmProjects/predicted_brain_age/outputs/homogeneous_dataset.csv', index=False)


if __name__ == "__main__":
    main()
