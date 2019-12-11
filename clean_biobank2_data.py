"""TEMPORARY SCRIPT to clean Biobank scanner 2 data.
This script may be incorporated into clean_biobank1_data script eventually.
The main difference is that the participants file seems to be in xls rather than tsv format for some reason.
Furthermore, not all Scanner2 data seem to be preprocessed yet"""

from pathlib import Path

import pandas as pd

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """Clean UK Biobank scanner 2 data."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner2'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'ukb22321.csv'
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'participants.xls'
    id_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'freesurferData.csv'

    output_ids_filename = 'cleaned_ids_scanner2_noqc.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(demographic_path, id_path)

    participants_df = pd.read_excel(participants_path, usecols=['Participant_ID', 'Diagn'])
    participants_df['ID'] = participants_df['Participant_ID'].str.split('-').str[1]
    participants_df['ID'] = pd.to_numeric(participants_df['ID'])

    dataset = pd.merge(dataset, participants_df, on='ID')

    # Exclude ages with <100 participants,
    dataset = dataset[dataset['Age'] > 46]

    # Exclude non-white ethnicities due to small subgroups
    dataset = dataset[dataset['Ethnicity'] == 'White']

    # Exclude patients
    dataset = dataset[dataset['Diagn'] == 1]

    output_ids_df = pd.DataFrame(dataset['Participant_ID'])
    output_ids_df.to_csv(experiment_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()