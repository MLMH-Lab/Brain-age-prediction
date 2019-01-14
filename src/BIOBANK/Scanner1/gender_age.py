"""
Script to explore distribution of age and gender in UK BIOBANK dataset from scanner1
Aim is to plot a line graph of age vs number of subjects for male/female

References
[1] - Zhao, Lu, et al. "Age-Related Differences in Brain Morphology and the Modifiers in Middle-Aged and Older Adults."
Cerebral Cortex (2018).

"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_save_histogram(male_ages, female_ages):
    """Create histogram of age distribution by gender and saves in output folder as png"""
    plt.style.use('seaborn-whitegrid')
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

    plt.savefig('../../../outputs/gender_age_dist_BIOBANK.png')
    plt.show()


def display_demographics(dataset_excl_nan):
    """Print and save demographic information about sample size and age"""
    male_code = 1
    female_code = 0

    n_total = len(dataset_excl_nan)
    n_male = len(dataset_excl_nan.groupby('Gender').get_group(male_code))
    n_female = len(dataset_excl_nan.groupby('Gender').get_group(female_code))

    age_mean = dataset_excl_nan.Age.mean()
    age_sd = dataset_excl_nan.Age.std()
    age_range_min = dataset_excl_nan.Age.min()
    age_range_max = dataset_excl_nan.Age.max()

    age_male_mean = dataset_excl_nan.groupby('Gender').get_group(male_code).Age.mean()
    age_female_mean = dataset_excl_nan.groupby('Gender').get_group(female_code).Age.mean()
    age_male_sd = dataset_excl_nan.groupby('Gender').get_group(male_code).Age.std()
    age_female_sd = dataset_excl_nan.groupby('Gender').get_group(female_code).Age.std()
    age_male_range_min = dataset_excl_nan.groupby('Gender').get_group(male_code).Age.min()
    age_female_range_min = dataset_excl_nan.groupby('Gender').get_group(female_code).Age.min()
    age_male_range_max = dataset_excl_nan.groupby('Gender').get_group(male_code).age.max()
    age_female_range_max = dataset_excl_nan.groupby('Gender').get_group(female_code).Age.max()

    print("Sample sizes [Total (m / f)]: %d (%d / %d)" % (n_total, n_male, n_female))
    print("Age - total [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f"
          % (age_mean, age_sd, age_range_min, age_range_max))
    print("Age - male [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f"
          % (age_male_mean, age_male_sd, age_male_range_min, age_male_range_max))
    print("Age - female [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f"
          % (age_female_mean, age_female_sd, age_female_range_min, age_female_range_max))

    # Saves into txt file
    file = open('../../../outputs/demographics.txt', 'w')

    file.write("Demographics for UK BIOBANK Scanner1 \n \n")
    file.write("Sample sizes [Total (m / f)]: %d (%d / %d) \n" % (n_total, n_male, n_female))
    file.write("Age - total [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f \n"
               % (age_mean, age_sd, age_range_min, age_range_max))
    file.write("Age - male [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f \n"
               % (age_male_mean, age_male_sd, age_male_range_min, age_male_range_max))
    file.write("Age - female [mean (SD), min-max]: %5.2f (%5.2f), %5.2f-%5.2f \n"
               % (age_female_mean, age_female_sd, age_female_range_min, age_female_range_max))
    file.close()


def main():
    dataset_demographic = pd.read_csv('../../../data/BIOBANK/Scanner1/participants.tsv', sep='\t')
    dataset_excl_nan = dataset_demographic.dropna()

    # Histogram of age distribution by gender
    male_code = 1
    female_code = 0

    male_ages = dataset_excl_nan.groupby('Gender').get_group(male_code).Age
    female_ages = dataset_excl_nan.groupby('Gender').get_group(female_code).Age

    plot_save_histogram(male_ages, female_ages)

    """Script to look at demographics info based on Zhao et al 2018
    Required are: number of subjects, gender split, age range,"""

    display_demographics(dataset_excl_nan)


if __name__ == "__main__":
    main()
