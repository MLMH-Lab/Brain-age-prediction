"""
Script to explore distribution of education with regards to age in UK BIOBANK dataset from scanner1
Aim is to plot a histogram of age vs number of subjects for education level
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def plot_save_histogram_education(edu_level_1, edu_level_2, edu_level_3, edu_level_4):
    """Create histogram of age distribution by education level and saves in output folder as png"""
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 7))

    plt.hist(edu_level_1, color='blue', histtype='step', lw=2, bins=range(45, 75, 1), label='GCSE')
    plt.hist(edu_level_2, color='red', histtype='step', lw=2, bins=range(45, 75, 1), label='A levels')
    plt.hist(edu_level_3, color='yellow', histtype='step', lw=2, bins=range(45, 75, 1),
             label='Professional qualification')
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
    # Define what subjects were modeled: total, male or female
    subjects = 'total'

    # Loading dataset with age and highest education level
    dataset = pd.read_csv(PROJECT_ROOT / 'outputs' / subjects / 'age_predictions_demographics.csv')
    dataset = dataset.dropna(subset=['Education_highest'])

    # Histogram of age distribution by education level
    gcse_code = 1
    alevels_code = 2
    prof_qual_code = 3
    uni_code = 4

    gcse_ages = dataset.groupby('Education_highest').get_group(gcse_code).Age
    alevels_ages = dataset.groupby('Education_highest').get_group(alevels_code).Age
    prof_qual_ages = dataset.groupby('Education_highest').get_group(prof_qual_code).Age
    uni_ages = dataset.groupby('Education_highest').get_group(uni_code).Age

    plot_save_histogram_education(gcse_ages, alevels_ages, prof_qual_ages, uni_ages)


if __name__ == "__main__":
    main()
