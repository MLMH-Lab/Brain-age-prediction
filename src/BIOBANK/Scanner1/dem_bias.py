"""Script to assess potential bias in confounding factors: gender vs age, ethnicity vs age;
Supplementary data and labels acquired from https://biobank.ctsu.ox.ac.uk/crystal/search.cgi"""

import pandas as pd


def main():

    # Loading supplementary demographic data
    dataset_demographic_suppl = pd.read_csv(
        '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/ukb22321.csv',
        usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    dataset_demographic_suppl.columns = ['ID', 'Gender', 'Ethnicity', 'Age at MRI']
    dataset_demographic_suppl_excl_nan = dataset_demographic_suppl.dropna()


if __name__ == "__main__":
    main()
