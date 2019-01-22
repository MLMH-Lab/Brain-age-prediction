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


if __name__ == "__main__":
    main()
