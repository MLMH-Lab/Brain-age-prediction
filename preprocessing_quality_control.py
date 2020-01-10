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

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()
parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')
parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')
parser.add_argument('-M', '--mriqc_threshold',
                    dest='mriqc_threshold',
                    nargs='?',
                    type=float, default=0.5,
                    help='Threshold value for MRIQC.')
parser.add_argument('-Q', '--qoala_threshold',
                    dest='qoala_threshold',
                    nargs='?',
                    type=float, default=0.5,
                    help='Threshold value for Qoala.')
args = parser.parse_args()


def main(experiment_name, scanner_name, mriqc_threshold, qoala_threshold):
    """Remove UK Biobank participants that did not pass quality checks."""
    # ----------------------------------------------------------------------------------------
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids_noqc.csv'
    mriqc_prob_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'mriqc_prob.csv'
    qoala_prob_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'qoala_prob.csv'

    qc_output_filename = 'cleaned_ids.csv'

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    ids_df = pd.read_csv(ids_path, sep='\t')
    prob_mriqc_df = pd.read_csv(mriqc_prob_path)
    prob_qoala_df = pd.read_csv(qoala_prob_path)

    prob_mriqc_df = prob_mriqc_df.rename(columns={'prob_y': 'mriqc_prob'})
    prob_mriqc_df['Image_ID'] = prob_mriqc_df['subject_id'] + '_ses-bl_T1w/'
    prob_mriqc_df = prob_mriqc_df[['Image_ID', 'mriqc_prob']]

    prob_qoala_df = prob_qoala_df.rename(columns={'image_id': 'Image_ID', 'prob_qoala': 'qoala_prob'})
    prob_qoala_df = prob_qoala_df[['Image_ID', 'qoala_prob']]

    qc_df = pd.merge(prob_mriqc_df, prob_qoala_df, on='Image_ID')

    selected_subjects = qc_df[(qc_df['mriqc_prob'] < mriqc_threshold) | (qc_df['qoala_prob'] < qoala_threshold)]

    ids_qc_df = pd.merge(ids_df, selected_subjects[['Image_ID']], on='Image_ID')

    ids_qc_df.to_csv(experiment_dir / qc_output_filename, index=False)


if __name__ == "__main__":
    main(args.experiment_name, args.scanner_name,
         args.mriqc_threshold, args.qoala_threshold)
