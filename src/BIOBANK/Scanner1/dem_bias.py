"""Script to assess potential bias in confounding factors: gender vs age, ethnicity vs age;
Supplementary data and labels acquired from https://biobank.ctsu.ox.ac.uk/crystal/search.cgi"""

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


def chi2_test(df, gender):
    """Perform Pearson chi-squared test for gender distribution per age"""

    ages_fre = df['Age'].value_counts()
    gender_expected_per_age = ages_fre / 2
    gender_observed_overview = pd.crosstab(df['Gender'], df['Age']).transpose()
    gender_observed_per_age = gender_observed_overview[gender]

    chi2, p = stats.chisquare(gender_observed_per_age, gender_expected_per_age)
    msg = "Chi-square test for: {}\nTest Statistic: {}\np-value: {}"
    print(msg.format(gender, chi2, p))


def chi2_contingency_test(crosstab_df, age_combinations, age1, age2):
    """Perform multiple 2x2 Pearson chi-square analyses"""

    cont_table = crosstab_df[[age1, age2]]
    chi2, p, dof, expected = stats.chi2_contingency(cont_table, correction=False)
    # msg = "Chi-square test for ages {} vs {}\nTest Statistic: {}\np-value: {}"
    # print(msg.format(age1, age2, chi2, p))

    # Bonferroni correction for multiple comparisons
    sig_level = 0.05 / len(age_combinations)
    msg = "Chi-square test for ages {} vs {} is significant:\nTest Statistic: {}\np-value: {}"
    if p < sig_level:
        print(msg.format(age1, age2, chi2, p))


# test chi2_contingency_test function
chi2_contingency_test(gender_observed, age_combinations, 47.0, 48.0)



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

    # Exclude ages with <100 participants, exclude non-white ethnicities due to small subgroups
    dataset_dem_ab46 = dataset_dem_excl_nan_grouped[dataset_dem_excl_nan_grouped['Age'] > 46]
    dataset_dem_ab46_ethn = dataset_dem_ab46[dataset_dem_ab46['Ethnicity'] == 'White']

    # Export ethnicity distribution
    fre_table(dataset_dem_ab46_ethn, 'Ethnicity')

    chi2_test(dataset_dem_ab46_ethn, 'Female')
    chi2_test(dataset_dem_ab46_ethn, 'Male')

    # Perform chi2 contingency analysis for each age combination
    gender_observed = pd.crosstab(dataset_dem_ab46_ethn['Gender'], dataset_dem_ab46_ethn['Age'])
    age_list = list(gender_observed.columns)
    age_combinations = list(itertools.product(age_list, age_list))
    age_combinations_new = []
    for age_tuple in age_combinations:
        if (age_tuple[1], age_tuple[0]) not in age_combinations_new:
            if age_tuple[0] != age_tuple[1]:
                age_combinations_new.append(age_tuple)

    for age_tuple in age_combinations:
        chi2_contingency_test(gender_observed, age_tuple[0], age_tuple[1])


if __name__ == "__main__":
    main()


