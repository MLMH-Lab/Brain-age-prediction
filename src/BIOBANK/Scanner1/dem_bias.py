"""Script to assess potential bias in confounding factors: gender vs age, ethnicity vs age;
Supplementary data and labels acquired from https://biobank.ctsu.ox.ac.uk/crystal/search.cgi"""

import pandas as pd
import matplotlib.pyplot as plt


def fre_plot(df, col_name):
    """Frequency plot of a dataframe column to visually assess distribution"""

    df[col_name].value_counts().plot('bar')
    plt.show()

# Test fre_plot function
fre_plot(dataset_dem_excl_nan, 'Gender')
fre_plot(dataset_dem_excl_nan, 'Ethnicity')


def main():

    # Loading supplementary demographic data
    dataset_dem = pd.read_csv(
        '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/ukb22321.csv',
        usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    dataset_dem.columns = ['ID', 'Gender', 'Ethnicity', 'Age']
    dataset_dem_excl_nan = dataset_dem.dropna()

    ethnicity_dict = {
        1: 'White',
        1001: 'British',
        2001: 'White and Black Caribbean',
        3001: 'Indian',
        4001: 'Caribbean',
        2: 'Mixed',
        1002: 'Irish',
        2002: 'White and Black African',
        3002: 'Pakistani',
        4002: 'African',
        3: 'Asian or Asian British',
        1003: 'Any other white background',
        2003: 'White and Asian',
        3003: 'Bangladeshi',
        4003: 'Any other Black background',
        4: 'Black or Black British',
        2004: 'Any other mixed background',
        3004: 'Any other Asian background',
        5: 'Chinese',
        6: 'Other ethnic group',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }

    grouped_ethnicity_dict = {
        1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
        2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
        3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
        4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
        5: 'Chinese',
        6: 'Other',
        -1: 'Not known', -3: 'Not known'
    }

    dataset_dem_excl_nan = dataset_dem_excl_nan.replace({'Ethnicity': ethnicity_dict})
    dataset_dem_excl_nan_grouped = dataset_dem_excl_nan.replace({'Ethnicity': grouped_ethnicity_dict})


if __name__ == "__main__":
    main()


