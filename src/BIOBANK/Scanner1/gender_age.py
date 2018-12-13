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


def age_dict(df, x, direction):
    '''Creates dictionary with number of subjects per age based on df col Age'''
    age_dict_x = {}
    for label, row in df.iterrows():
        age = row["Age"]
        if direction == '>':
            if age > x:
                if age in age_dict_x:
                    age_dict_x[age] += 1
                else:
                    age_dict_x[age] = 1
        elif direction == '<':
            if age < x:
                if age in age_dict_x:
                    age_dict_x[age] += 1
                else:
                    age_dict_x[age] = 1
        else:
            print("direction parameter has to be string < or >")
    print(age_dict_x)


# create dataset excl subjects with missing data and including ages 49-75 (see Zhao et al 2018)
age_dict(DATASET_EXCL_NAN, 50, '<')
age_dict(DATASET_EXCL_NAN, 70, '>')
DATASET_AGES_49 = DATASET_EXCL_NAN[DATASET_EXCL_NAN["Age"] > 48]
DATASET_AGES_49_75 = DATASET_AGES_49[DATASET_AGES_49["Age"] < 76]
print(len(DATASET_EXCL_NAN))
print(len(DATASET_AGES_49))
print(len(DATASET_AGES_49_75))
print(DATASET_AGES_49_75[["Participant_ID"]])

# quick way to display difference using pandas plots
DATASET_EXCL_NAN.groupby('Gender').Age.hist(bins=range(49, 80, 1), alpha=0.5, histtype='step', lw=5)
DATASET_AGES_50_75.groupby('Gender').Age.hist(bins=range(49, 80, 1), alpha=0.5, histtype='step', lw=5)

male_ages = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age
female_ages = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age
plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=range(47, 79, 1), label='male')
plt.hist(female_ages, color='red', histtype='step', lw=2, bins=range(47, 79, 1), label='female')

plt.title("Age distribution in UK BIOBANK")
plt.axis('tight')
plt.xlabel("Age [years]")
plt.ylabel("Number of subjects")
plt.legend(loc='upper right')
plt.tick_params(labelsize=18)
fig = plt.figure(figsize=(25, 15))
plt.show(fig)

plt.savefig('gender_age_dist_BIOBANK.png')


# def main():
#    dataset_demographic_filename = 2
#    pass

# if __name__ == "__main__":
#    main()
