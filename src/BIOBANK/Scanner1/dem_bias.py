"""Script to assess sample homogeneity in UK BIOBANK Scanner1: gender vs age, ethnicity vs age;
Supplementary data and labels acquired from https://biobank.ctsu.ox.ac.uk/crystal/search.cgi

Step 1: Organising dataset
Step 2: Visualisation of distribution
Step 3: Chi-square contingency analysis
Step 4: Remove subjects based on chi-square results to achieve homogeneous sample in terms of gender and ethnicity"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools


def fre_plot_split(df, col_name1, col_name2):
    """Frequency plot of df column1 grouped by column2"""

    pd.crosstab(df[col_name1], df[col_name2]).plot.bar()
    plt.show()


def fre_table(df, col_name):
    """Export frequency table of column as csv"""

    fre_table = df[col_name].value_counts()
    file_name = col_name + '_fre_table.csv'
    fre_table.to_csv('/home/lea/PycharmProjects/predicted_brain_age/outputs/' + file_name)


def chi2_contingency_test(crosstab_df, age_combinations, sig_list, age1, age2):
    """Perform multiple 2x2 Pearson chi-square analyses, corrected for multiple comparisons"""

    contingency_table = crosstab_df[[age1, age2]]
    chi2, p, dof, expected = stats.chi2_contingency(contignency_table, correction=False)

    # Bonferroni correction for multiple comparisons; use sig_list to check which ages are most different
    sig_level = 0.05 / len(age_combinations)
    msg = "Chi-square test for ages {} vs {} is significant:\nTest Statistic: {}\np-value: {}\n"
    if p < sig_level:
        sig_list.append(age1)
        sig_list.append(age2)
        print(msg.format(age1, age2, chi2, p))


def get_ids_to_drop(df, age, gender, n_to_drop, id_list):
    """Extract random sample of participant IDs per age per gender to drop from total sample"""

    df_filter1 = df[df['Age'] == age]
    df_filter2 = df_filter1[df_filter1['Gender'] == gender]

    # random sample of IDs to drop
    df_to_drop = df_filter2.sample(n_to_drop)
    id_list.append(list(df_to_drop['ID']))

    return id_list


def main():

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
    fre_table(dataset_dem_excl_nan_grouped, 'Ethnicity')
    fre_table(dataset_dem_excl_nan_grouped, 'Age')

    # Exclude ages with <100 participants, exclude non-white ethnicities due to small subgroups
    dataset_dem_ab46 = dataset_dem_excl_nan_grouped[dataset_dem_excl_nan_grouped['Age'] > 46]
    dataset_dem_ab46_ethn = dataset_dem_ab46[dataset_dem_ab46['Ethnicity'] == 'White']

    # Create list of unique age combinations for chi2 contingency analysis
    gender_observed = pd.crosstab(dataset_dem_ab46_ethn['Gender'], dataset_dem_ab46_ethn['Age'])
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

    # Obtain gender proportion per age group
    gender_prop = gender_observed.transpose()
    gender_prop['proportion'] = gender_prop['Female'] / gender_prop['Male']

    # Undersample the more prominent gender per age in dict_sig and store removed IDs in ids_to_drop
    ids_to_drop = []

    for key in dict_sig.keys():

        if dict_sig[key] > 2:
            if gender_observed.loc['Female', key] > gender_observed.loc['Male', key]:
                gender_higher = 'Female'
            elif gender_observed.loc['Female', key] < gender_observed.loc['Male', key]:
                gender_higher = 'Male'
            else:
                print("Error with: " + str(key))

            gender_diff = gender_observed.loc['Female', key] - gender_observed.loc['Male', key]
            diff_to_remove = int(abs(gender_diff))

            get_ids_to_drop(dataset_dem_ab46_ethn, key, gender_higher, diff_to_remove, ids_to_drop)

    # Flatten ids_to_drop because it is a list of lists
    flattened_ids_to_drop = []
    for id_list in ids_to_drop:
        for id in id_list:
            flattened_ids_to_drop.append(id)

    # Create new dataset with ids from flattened_ids_to_drop removed
    reduced_dataset = dataset_dem_ab46_ethn[~dataset_dem_ab46_ethn.ID.isin(flattened_ids_to_drop)]

    # Perform chi2 contingency analysis again on reduced dataset to check if gender proportion are homogeneous
    # Same code as above with new variables
    gender_observed_2 = pd.crosstab(reduced_dataset['Gender'], reduced_dataset['Age'])
    age_list_2 = list(gender_observed_2.columns)
    age_combinations_2 = list(itertools.product(age_list_2, age_list_2))
    age_combinations_new_2 = []
    for age_tuple in age_combinations_2:
        if (age_tuple[1], age_tuple[0]) not in age_combinations_new_2:
            if age_tuple[0] != age_tuple[1]:
                age_combinations_new_2.append(age_tuple)

    sig_list_2 = []
    for age_tuple in age_combinations_new_2:
        chi2_contingency_test(gender_observed_2, age_combinations_new_2, sig_list_2, age_tuple[0], age_tuple[1])

    dict_sig_2 = {}
    for item in sig_list_2:
        if item in dict_sig_2:
            dict_sig_2[item] += 1
        elif item not in dict_sig_2:
            dict_sig_2[item] = 1
        else:
            print("error with " + str(item))


if __name__ == "__main__":
    main()


