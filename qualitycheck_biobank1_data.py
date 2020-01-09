#!/usr/bin/env python3
"""Remove bad quality data.

This script removes UK Biobank Scanner 1 participants that did not pass the quality check of raw MRI data and Freesurfer segmentation

"""

from pathlib import Path

import pandas as pd

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """Remove UK Biobank participants that did not pass quality checks."""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids_noqc.csv'
    qc_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'BIOBANK_QC.csv'

    qc_output_ids_filename = 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    dataset_clean = load_demographic_data(demographic_path, id_path)
    dataset_qc = load_demographic_data(demographic_path, qc_path)
    dataset_qc_include = dataset_qc[dataset_qc['my_suggestion_exclude']==False]

    dataset_clean_qc = pd.merge(dataset_clean, dataset_qc_include, on='Participant_ID')

    qc_output_ids_df = pd.DataFrame(dataset_clean_qc['Participant_ID'])
    qc_output_ids_df.to_csv(experiment_dir / qc_output_ids_filename, index=False)


if __name__ == "__main__":
    main()
