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
fre_plot(dataset_dem_excl_nan_grouped, 'Ethnicity')


def fre_plot_split(df, col_name1, col_name2):
    """Frequency plot of df column1 grouped by column2"""

    pd.crosstab(df[col_name1], df[col_name2]).plot.bar()
    plt.show()

# Test fre_plot_split function
fre_plot_split(dataset_dem_excl_nan_grouped, 'Ethnicity', 'Gender')


def fre_table(df, col_name):
    """Export frequency table of column as csv"""

    fre_table = df[col_name].value_counts()
    file_name = col_name + '_fre_table.csv'
    fre_table.to_csv('/home/lea/PycharmProjects/predicted_brain_age/outputs/' + file_name)

# Test fre_plot_split function
fre_table(dataset_dem_excl_nan_grouped, 'Ethnicity')


def main():

    # Loading supplementary demographic data
    dataset_dem = pd.read_csv(
        '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/ukb22321.csv',
        usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    dataset_dem.columns = ['ID', 'Gender', 'Ethnicity', 'Age']
    dataset_dem_excl_nan = dataset_dem.dropna()

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




if __name__ == "__main__":
    main()


