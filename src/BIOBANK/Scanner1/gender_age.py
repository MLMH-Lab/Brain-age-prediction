"""
Script to explore distribution of age and gender in UK BIOBANK dataset from scanner1
Aim is to plot a line graph of age vs number of subjects for male/female
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_DEMOGRAPHIC_FILENAME = '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/participants.tsv'
DATASET_AVAILABLE = pd.read_csv(DATASET_DEMOGRAPHIC_FILENAME, sep='\t')
DATASET_EXCL_NAN = DATASET_AVAILABLE.dropna()
DATASET_MALE = DATASET_EXCL_NAN[DATASET_EXCL_NAN["Gender"] == 1]
DATASET_FEMALE = DATASET_EXCL_NAN[DATASET_EXCL_NAN["Gender"] == 0]

# create dataset excl subjects with missing data and including ages 49-75
DATASET_AGES_49_74 = DATASET_EXCL_NAN[(DATASET_EXCL_NAN.Age > 48) & (DATASET_EXCL_NAN.Age < 74)]

# quick way to display difference using pandas plots
DATASET_EXCL_NAN.groupby('Gender').Age.hist(bins=range(45, 80, 1), alpha=0.5, histtype='step', lw=2)
DATASET_AGES_49_74.groupby('Gender').Age.hist(bins=range(49, 74, 1), alpha=0.5, histtype='step', lw=2)
DATASET_AGES_49_74.Age.hist()

# more detailed display, select all to run
plt.style.use('seaborn-whitegrid')

male_ages = DATASET_AGES_49_74.groupby('Gender').get_group(1).Age
female_ages = DATASET_AGES_49_74.groupby('Gender').get_group(0).Age

plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=range(47, 79, 1), label='male')
plt.hist(female_ages, color='red', histtype='step', lw=2, bins=range(47, 79, 1), label='female')

plt.title("Age distribution in UK BIOBANK")
plt.axis('tight')
plt.xlabel("Age [years]")
plt.ylabel("Number of subjects")
plt.legend(loc='upper right')
plt.tick_params(labelsize=10)
fig = plt.figure(figsize=(25, 15))
plt.show(fig)

plt.savefig('gender_age_dist_BIOBANK.png')

"""Script to look at demographics info based on Zhao et al 2018
Required are: number of subjects, gender split, age range,"""
