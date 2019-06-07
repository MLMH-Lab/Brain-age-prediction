"""Clean UK Biobank scanner 1 (Cheadle) data.

Subjects from the Assessment Centre from Cheadle (code 11025) are majority white.
Besides, some ages have very low number of subjects (<100). The ethnics minorities
and age with low number are remove from further analysis as well subjects with any
mental or brain disorder.

"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()


def main():
    """Clean UK Biobank scanner 1 data."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants_scanner1.tsv'
    id_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'

    output_ids_filename = 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Loading supplementary demographic data
    demographic_df = pd.read_csv(demographic_path, usecols=['eid', '31-0.0', '21003-2.0', '21000-0.0'])
    demographic_df.columns = ['ID', 'Gender', 'Ethnicity', 'Age']
    demographic_df = demographic_df.dropna()

    id_df = pd.read_csv(id_path)
    # Create a new 'ID' column to match supplementary demographic data
    if 'Participant_ID' in id_df.columns:
        # For create_homogeneous_data.py output
        id_df['ID'] = id_df['Image_ID'].str.split('-').str[1]
    else:
        # For freesurferData dataframe
        id_df['ID'] = id_df['Image_ID'].str.split('_').str[0]
        id_df['ID'] = id_df['ID'].str.split('-').str[1]

    id_df['ID'] = pd.to_numeric(id_df['ID'])

    # Merge supplementary demographic data with ids
    dataset = pd.merge(id_df, demographic_df, on='ID')

    participants_df = pd.read_csv(participants_path, sep='\t', usecols=['Participant_ID', 'Diagn'])
    participants_df['ID'] = participants_df['Participant_ID'].str.split('-').str[1]
    participants_df['ID'] = pd.to_numeric(participants_df['ID'])

    dataset = pd.merge(dataset, participants_df, on='ID')

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

    dataset = dataset.replace({'Gender': gender_dict})
    dataset = dataset.replace({'Ethnicity': ethnicity_dict})

    # Exclude ages with <100 participants,
    dataset = dataset[dataset['Age'] > 46]

    # Exclude non-white ethnicities due to small subgroups
    dataset = dataset[dataset['Ethnicity'] == 'White']

    # Exclude patients
    dataset = dataset[dataset['Diagn'] == 1]

    output_ids_df = pd.DataFrame(dataset['Participant_ID'])
    output_ids_df.to_csv(experiment_dir / output_ids_filename, index=False)
