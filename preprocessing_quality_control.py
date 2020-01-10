#!/usr/bin/env python3
"""Perform quality control.

This script removes participants that did not pass the quality
control performed using MRIQC [1] and Qoala [2].

References
----------
[1] - Esteban, Oscar, et al. "MRIQC: Advancing the automatic prediction
of image quality in MRI from unseen sites." PloS one 12.9 (2017): e0184661.

[2] - Klapwijk, Eduard T., et al. "Qoala-T: A supervised-learning tool for
quality control of FreeSurfer segmented MRI data." NeuroImage 189 (2019): 116-129.
"""
import argparse
from pathlib import Path

import pandas as pd

from utils import load_demographic_data


PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()
parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')
parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')
args = parser.parse_args()


def main(experiment_name, scanner_name, mriqc_threshold=0.5, qoala_threshold=0.5):
    """Remove UK Biobank participants that did not pass quality checks."""
    # ----------------------------------------------------------------------------------------
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids_noqc.csv'
    qc_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'BIOBANK_QC.csv'

    qc_output_ids_filename = 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    dataset_clean = load_demographic_data(demographic_path, id_path)
    dataset_qc = load_demographic_data(demographic_path, qc_path)
    dataset_qc_include = dataset_qc[dataset_qc['my_suggestion_exclude'] == False]

    dataset_clean_qc = pd.merge(dataset_clean, dataset_qc_include, on='participant_id')

    qc_output_ids_df = pd.DataFrame(dataset_clean_qc['participant_id'])
    qc_output_ids_df.to_csv(experiment_dir / qc_output_ids_filename, index=False)


if __name__ == "__main__":
    main(args.experiment_name, args.scanner_name)
