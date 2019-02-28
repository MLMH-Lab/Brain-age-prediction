"""
Script to explore distribution of gender and education with regards to age in UK BIOBANK dataset from scanner1
Aim is to plot a line graph of age vs number of subjects for male/female + age vs number of subjects for education level

References
[1] - Zhao, Lu, et al. "Age-Related Differences in Brain Morphology and the Modifiers in Middle-Aged and Older Adults."
Cerebral Cortex (2018).

"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path('../../../') # Todo: Improve before publish code


def plot_save_histogram(male_ages, female_ages):
    """Create histogram of age distribution by gender and saves in output folder as png"""
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 7))

    plt.hist(male_ages, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='male')
    plt.hist(female_ages, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='female')

    plt.title("Age distribution in UK BIOBANK by gender", fontsize=17)
    plt.axis('tight')
    plt.xlabel("Age at MRI scan [years]", fontsize=15)
    plt.ylabel("Number of subjects", fontsize=15)
    plt.xticks(range(45, 75, 5))
    plt.yticks(range(0, 401, 50))
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)

    output_img_path = PROJECT_ROOT / 'outputs' / 'gender_age_dist_BIOBANK.png'
    plt.savefig(str(output_img_path))
    plt.show()


def plot_save_histogram_education(edu_level_1, edu_level_2, edu_level_3, edu_level_4):
    """Create histogram of age distribution by education level and saves in output folder as png"""
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 7))

    plt.hist(edu_level_1, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='GCSE')
    plt.hist(edu_level_2, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='A levels')
    plt.hist(edu_level_3, color='yellow', histtype='step', lw=2, bins=range(45, 75, 1), label='Professional qualification')
    plt.hist(edu_level_4, color='green', histtype='step', lw=2, bins=range(45, 75, 1), label='University')

    plt.title("Age distribution in UK BIOBANK by education level", fontsize=17)
    plt.axis('tight')
    plt.xlabel("Age at MRI scan [years]", fontsize=15)
    plt.ylabel("Number of subjects", fontsize=15)
    plt.xticks(range(45, 75, 5))
    plt.yticks(range(0, 401, 50))
    plt.legend(loc='upper right', fontsize=13)
    plt.tick_params(labelsize=13)

    output_img_path = PROJECT_ROOT / 'outputs' / 'education_age_dist_BIOBANK.png'
    plt.savefig(str(output_img_path))
    plt.show()


def main():
    # Loading data
    dataset_demographic = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv', sep='\t')
    dataset_excl_nan = dataset_demographic.dropna()

    # Histogram of age distribution by gender
    male_code = 1
    female_code = 0

    male_ages = dataset_excl_nan.groupby('Gender').get_group(male_code).Age
    female_ages = dataset_excl_nan.groupby('Gender').get_group(female_code).Age

    plot_save_histogram(male_ages, female_ages)

    # Histogram of age distribution by education level
    gcse_code = 1
    alevels_code = 2
    prof_qual_code = 3
    uni_code = 4

    gcse_ages = dataset_excl_nan.groupby('Education_highest').get_group(gcse_code).Age
    alevels_ages = dataset_excl_nan.groupby('Education_highest').get_group(alevels_code).Age
    prof_qual_ages = dataset_excl_nan.groupby('Education_highest').get_group(prof_qual_code).Age
    uni_ages = dataset_excl_nan.groupby('Education_highest').get_group(uni_code).Age

    plot_save_histogram_education(gcse_ages, alevels_ages, prof_qual_ages, uni_ages)

    # Script to look at demographics info based on Zhao et al 2018
    # Required are: number of subjects, gender split, age range
    print('Whole dataset')
    print(dataset_excl_nan.Age.describe())
    dataset_excl_nan.Age.describe().to_csv(PROJECT_ROOT / 'outputs' / 'scanner01_whole_dataset_dem.csv')

    print('Grouped dataset')
    print(dataset_excl_nan.groupby('Gender').Age.describe())
    dataset_excl_nan.groupby('Gender').Age.describe().to_csv(PROJECT_ROOT / 'outputs' / 'scanner01_grouped_dem.csv')


if __name__ == "__main__":
    main()
