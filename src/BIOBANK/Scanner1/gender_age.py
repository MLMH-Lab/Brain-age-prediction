"""
Script to explore distribution of age and gender in UK BIOBANK dataset from scanner1
Aim is to plot a line graph of age vs number of subjects for male/female
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_save_histogram(male_ages, female_ages):
    """Creates histogram of age distribution by gender and saves in output folder as png"""
    plt.figure(figsize=(10, 7))
    plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='male')
    plt.hist(female_ages, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='female')
    plt.title("Age distribution in UK BIOBANK", fontsize=17)
    plt.axis('tight')
    plt.xlabel("Age at MRI scan [years]", fontsize=15)
    plt.ylabel("Number of subjects", fontsize=15)
    plt.xticks(range(45, 75, 5))
    plt.yticks(range(0, 401, 50))
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)
    plt.savefig('/home/lea/PycharmProjects/predicted_brain_age/outputs/gender_age_dist_BIOBANK.png')
    plt.show()


def display_demographics():
    """Prints demographic information about sample size and age"""
    print("Sample sizes [Total (m / f)]: %d (%d / %d)" % (N_TOTAL, N_MALE, N_FEMALE))
    print("Age - total [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f"
          % (AGE_MEAN, AGE_SD, AGE_RANGE_MIN, AGE_RANGE_MAX))
    print("Age - male [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f"
          % (AGE_MALE_MEAN, AGE_MALE_SD, AGE_MALE_RANGE_MIN, AGE_MALE_RANGE_MAX))
    print("Age - female [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f"
          % (AGE_FEMALE_MEAN, AGE_FEMALE_SD, AGE_FEMALE_RANGE_MIN, AGE_FEMALE_RANGE_MAX))


def create_dem_file():
    """Saves display_demographics info to a txt file in output folder"""
    file = open('/home/lea/PycharmProjects/predicted_brain_age/outputs/demographics.txt', 'w')
    file.write("Demographics for UK BIOBANK Scanner1 \n \n")
    file.write("Sample sizes [Total (m / f)]: %d (%d / %d) \n" % (N_TOTAL, N_MALE, N_FEMALE))
    file.write("Age - total [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f \n"
               % (AGE_MEAN, AGE_SD, AGE_RANGE_MIN, AGE_RANGE_MAX))
    file.write("Age - male [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f \n"
               % (AGE_MALE_MEAN, AGE_MALE_SD, AGE_MALE_RANGE_MIN, AGE_MALE_RANGE_MAX))
    file.write("Age - female [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f \n"
               % (AGE_FEMALE_MEAN, AGE_FEMALE_SD, AGE_FEMALE_RANGE_MIN, AGE_FEMALE_RANGE_MAX))
    file.close()
    return file


def main():
    DATASET_DEMOGRAPHIC = pd.read_csv('/home/lea/PycharmProjects/'
                                      'redicted_brain_age/data/BIOBANK/Scanner1/participants.tsv', sep='\t')
    DATASET_EXCL_NAN = DATASET_DEMOGRAPHIC.dropna()

    # histogram of age disttibution by gender
    plt.style.use('seaborn-whitegrid')

    male_ages = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age
    female_ages = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age

    plot_save_histogram(male_ages, female_ages)

    """Script to look at demographics info based on Zhao et al 2018
    Required are: number of subjects, gender split, age range,"""

    N_TOTAL = len(DATASET_EXCL_NAN)
    N_MALE = len(DATASET_EXCL_NAN.groupby('Gender').get_group(1))
    N_FEMALE = len(DATASET_EXCL_NAN.groupby('Gender').get_group(0))

    AGE_MEAN = DATASET_EXCL_NAN.Age.mean()
    AGE_MEDIAN = DATASET_EXCL_NAN.Age.median()
    AGE_SD = DATASET_EXCL_NAN.Age.std()
    AGE_RANGE_MIN = DATASET_EXCL_NAN.Age.min()
    AGE_RANGE_MAX = DATASET_EXCL_NAN.Age.max()

    AGE_MALE_MEAN = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age.mean()
    AGE_FEMALE_MEAN = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age.mean()
    AGE_MALE_MEDIAN = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age.median()
    AGE_FEMALE_MEDIAN = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age.median()
    AGE_MALE_SD = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age.std()
    AGE_FEMALE_SD = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age.std()
    AGE_MALE_RANGE_MIN = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age.min()
    AGE_FEMALE_RANGE_MIN = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age.min()
    AGE_MALE_RANGE_MAX = DATASET_EXCL_NAN.groupby('Gender').get_group(1).Age.max()
    AGE_FEMALE_RANGE_MAX = DATASET_EXCL_NAN.groupby('Gender').get_group(0).Age.max()

    display_demographics()
    create_dem_file()


if __name__ == "__main__":
    main()
