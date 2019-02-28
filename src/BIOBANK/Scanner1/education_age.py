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

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


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

    # Access education information
    dataset_edu = pd.read_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'),
                              usecols=['eid',
                                       '6138-2.0', '6138-2.1', '6138-2.2', '6138-2.3', '6138-2.4'])
    dataset_edu.columns = ['ID',
                           'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5']

    # Create new education cols to simulate ordinal scale
    education_dict = {1: 4, 2: 2, 3: 1, 4: 1, 5: 3, 6: 3}
    dataset_dem['Education_1'] = dataset_dem['Education_1'].map(education_dict)
    dataset_dem['Education_2'] = dataset_dem['Education_2'].map(education_dict)
    dataset_dem['Education_3'] = dataset_dem['Education_3'].map(education_dict)
    dataset_dem['Education_4'] = dataset_dem['Education_4'].map(education_dict)
    dataset_dem['Education_5'] = dataset_dem['Education_5'].map(education_dict)

    # Create col for maximum of education level per respondent
    dataset_dem['Education_highest'] = dataset_dem[['Education_1', 'Education_2', 'Education_3',
                                                    'Education_4', 'Education_5']].apply(max, axis=1)

    # Merge age_pred and dataset_dem datasets
    dataset = pd.merge(age_pred, dataset_dem, on='ID')


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