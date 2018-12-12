"""
Script to explore distribution of age and gender in UK BIOBANK dataset from scanner1
Aim is to plot a line graph of age vs number of subjects for male/female using seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATASET_DEMOGRAPHIC_FILENAME = '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/participants.tsv'
DATASET_AVAILABLE = pd.read_csv(DATASET_DEMOGRAPHIC_FILENAME, sep='\t')
DATASET_EXCL_NAN = DATASET_AVAILABLE.dropna()
DATASET_MALE = DATASET_EXCL_NAN[DATASET_EXCL_NAN["Gender"]==1]
DATASET_FEMALE = DATASET_EXCL_NAN[DATASET_EXCL_NAN["Gender"]==0]

COL_AGE = DATASET_EXCL_NAN['Age']
COL_GENDER = DATASET_EXCL_NAN['Gender']

def age_ls(df):
    """Returns Age columns from dataframe as list"""
    age_list = []
    for label, row in df.iterrows():
        col_age = df.loc[label, "Age"]
        age_list.append(int(col_age))
    return age_list

AGE_LIST_FEMALE = age_ls(DATASET_FEMALE)
AGE_LIST_MALE = age_ls(DATASET_MALE)

def age_fre(list):
    """Returns number of instances per age as dict"""
    age_count = {}
    for age in list:
        if age in age_count:
            age_count[age] += 1
        elif age not in age_count:
            age_count[age] = 1
        else:
            print("error with" + age)
    return age_count

AGE_FRE_FEMALE = age_fre(AGE_LIST_FEMALE)
AGE_FRE_MALE = age_fre(AGE_LIST_MALE)

AGE_FRE_FEMALE_DF = pd.DataFrame.from_dict(AGE_FRE_FEMALE, orient='index', dtype=int, columns=["Count"])
AGE_FRE_MALE_DF = pd.DataFrame.from_dict(AGE_FRE_MALE, orient='index', dtype=int, columns=["Count"])
AGE_FRE_FEMALE_DF["Age"] = AGE_FRE_FEMALE_DF.index
AGE_FRE_MALE_DF["Age"] = AGE_FRE_MALE_DF.index

age_plot_female = sns.lineplot(x="Age", y="Count", data=AGE_FRE_FEMALE_DF, legend='full', label='female', color='red')
age_plot_male = sns.lineplot(x="Age", y="Count", data=AGE_FRE_MALE_DF, legend='full', label='male', color='blue')
plt.title("Age distribution in UK BIOBANK")
plt.axis('tight')
plt.xlabel("Age [years]")
plt.ylabel("Number of subjects")
plt.legend(loc= 'upper right')
plt.show()


#def main():
#    dataset_demographic_filename = 2
#    pass

#if __name__ == "__main__":
#    main()