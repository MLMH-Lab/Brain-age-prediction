#!/usr/bin/env python3
"""Clean UK Biobank scanner 1 (Cheadle) data.

Subjects from the Assessment Centre from Cheadle (code 11025) are majority white.
Besides, some ages have very low number of subjects (<100). The ethnics minorities
and age with low number are remove from further analysis as well subjects with any
mental or brain disorder.

"""
from pathlib import Path

import pandas as pd

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """Clean UK Biobank scanner 1 data."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    id_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'

    output_ids_filename = 'cleaned_ids_noqc.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(demographic_path, id_path)

    participants_df = pd.read_csv(participants_path, sep='\t', usecols=['Participant_ID', 'Diagn'])
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
