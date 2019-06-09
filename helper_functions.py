"""
Helper functions.
"""
import pandas as pd


def load_demographic_data(demographic_path, id_path):
    """Load dataset using selected ids."""

    if demographic_path.suffix == '.tsv':
        demographic_df = pd.read_csv(demographic_path, sep='\t')
    else:
        demographic_df = pd.read_csv(demographic_path)

    # if using UK Biobank supplementary data
    if 'eid' in demographic_df.columns:
        demographic_df = demographic_df[['eid', '31-0.0', '21000-0.0', '21003-2.0']]
        demographic_df.columns = ['ID', 'Gender', 'Ethnicity', 'Age']

        # Labeling data
        ethnicity_dict = {
            1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
            2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
            3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
            4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
            5: 'Chinese',
            6: 'Other',
            -1: 'Not known', -3: 'Not known'
        }

        gender_dict = {
            0: 'Female',
            1: 'Male'
        }

        demographic_df = demographic_df.replace({'Gender': gender_dict})
        demographic_df = demographic_df.replace({'Ethnicity': ethnicity_dict})

    # if using participants.tsv file
    else:
        demographic_df['ID'] = demographic_df['Participant_ID'].str.split('-').str[1]

    demographic_df['ID'] = pd.to_numeric(demographic_df['ID'])
    demographic_df = demographic_df.dropna()

    id_df = pd.read_csv(id_path)
    # Create a new 'ID' column to match supplementary demographic data
    if 'Participant_ID' in id_df.columns:
        # For create_homogeneous_data.py output
        id_df['ID'] = id_df['Participant_ID'].str.split('-').str[1]
    else:
        # For freesurferData dataframe
        id_df['ID'] = id_df['Image_ID'].str.split('_').str[0]
        id_df['ID'] = id_df['ID'].str.split('-').str[1]

    id_df['ID'] = pd.to_numeric(id_df['ID'])

    # Merge supplementary demographic data with ids
    dataset = pd.merge(id_df, demographic_df, on='ID')

    return dataset
