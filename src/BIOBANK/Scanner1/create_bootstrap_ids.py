"""Script to create gender-homogeneous lists of Image IDs to feed into create_h5_dataset script"""


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():
    # Load final homogeneous dataset with Image IDs, age and gender variables
    dataset = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData_total.h5', key='table')

    # Find range of ages in homogeneous dataset
    age_min = int(dataset['Age'].min()) # 47
    age_max = int(dataset['Age'].max()) # 73

    # Loop over ages
    for age in range(age_min, (age_max + 1)):

        # Empty list to append Image IDs to
        list_ids = []

        # Get dataset for specific age only
        age_group = dataset.groupby('Age').get_group(age)

        # Loop over genders (0: female, 1:male)
        for gender in range(2):
            gender_group = age_group.groupby('Gender').get_group(gender)
            random_id = gender_group['Image_ID'].sample(n=1, replace=True, random_state=age)
            # random_row = gender_group.sample(n=1, replace=True, random_state=47)
            # random_id = gender_group.iloc[random_row.index, 3]
            list_ids.append(random_id)

    # Save list_ids as csv
    df_ids = pd.DataFrame(list_ids).transpose()
    file_name = 'homogeneous_ids_age_' + str(age) + '.csv'
    df_ids.to_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / file_name), index=False)


if __name__ == "__main__":
    main()