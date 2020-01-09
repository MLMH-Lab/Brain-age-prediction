#!/usr/bin/env python3
"""Clean UK Biobank scanner 2 data."""
from pathlib import Path

import pandas as pd

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """Clean UK Biobank scanner 2 data."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'ukb22321.csv'
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'participants.tsv'
    id_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'freesurferData.csv'
    qoala_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'Qoala_T_predictions_SCANNER02.csv'
    mriqc_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'group_T1w.tsv'
    mriqc2_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner2' / 'BIOBANKS2-unseen_pred.csv'

    output_ids_filename = 'biobank_scanner2_cleaned_ids.csv'
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
    # ----------------------------------------------------------------------------------------
    # Clean data based on quality control
    # qoala_df = pd.read_csv(qoala_path, usecols=['image_id', 'prob_qoala'])
    # qoala_df = qoala_df.rename(columns={"image_id": "Image_ID"})
    # dataset = pd.merge(dataset, qoala_df, on='Image_ID')
    #
    # mriqc_df = pd.read_csv(mriqc_path, sep='\t', usecols=['bids_name'])
    # mriqc2_df = pd.read_csv(mriqc2_path, usecols=['subject_id', 'prob_y'])
    #
    # mriqc_data_df = pd.concat((mriqc_df, mriqc2_df), axis=1)
    # mriqc_data_df['Image_ID'] = mriqc_data_df['bids_name'] + '/'
    # mriqc_data_df = mriqc_data_df.rename(columns={"prob_y": "mriqc_prob"})
    #
    # dataset_df = pd.merge(dataset, mriqc_data_df, on='Image_ID')
    #
    # mriqc_threshold = 0.6
    # qoala_threshold = 0.4
    # selected_subjects = dataset_df[(dataset_df['mriqc_prob'] < mriqc_threshold) & (dataset_df['prob_qoala'] < qoala_threshold)]

    output_ids_df = pd.DataFrame(dataset['Participant_ID'])
    output_ids_df.to_csv(experiment_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()
