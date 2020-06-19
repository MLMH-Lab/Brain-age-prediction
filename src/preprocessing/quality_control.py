#!/usr/bin/env python3
"""Perform quality control.

This script removes participants that did not pass the quality
control performed using MRIQC [1] and Qoala [2].

In Qoala, higher numbers indicate a higher chance of being a high quality scan
(Source: https://qoala-t.shinyapps.io/qoala-t_app/).

In MRIQC, higher values indicates a higher probability of being from MRIQC's class 1 ('exclude')
(Source: https://github.com/poldracklab/mriqc/blob/98610ad7596b586966413b01d10f4eb68366a038/mriqc/classifier/helper.py)

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

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='cleaned_ids_noqc.csv',
                    help='Filename indicating the ids to be used.')

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


def main(experiment_name, scanner_name, input_ids_file, mriqc_threshold, qoala_threshold):
    """Remove UK Biobank participants that did not pass quality checks."""
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / input_ids_file
    mriqc_prob_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'mriqc_prob.csv'
    qoala_prob_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'qoala_prob.csv'

    qc_output_filename = 'cleaned_ids.csv'

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    ids_df = pd.read_csv(ids_path)
    prob_mriqc_df = pd.read_csv(mriqc_prob_path)
    prob_qoala_df = pd.read_csv(qoala_prob_path)

    prob_mriqc_df = prob_mriqc_df.rename(columns={'prob_y': 'mriqc_prob'})
    prob_mriqc_df = prob_mriqc_df[['image_id', 'mriqc_prob']]

    prob_qoala_df = prob_qoala_df.rename(columns={'prob_qoala': 'qoala_prob'})
    prob_qoala_df = prob_qoala_df[['image_id', 'qoala_prob']]

    qc_df = pd.merge(prob_mriqc_df, prob_qoala_df, on='image_id')

    selected_subjects = qc_df[(qc_df['mriqc_prob'] < mriqc_threshold) | (qc_df['qoala_prob'] < qoala_threshold)]

    ids_qc_df = pd.merge(ids_df, selected_subjects[['image_id']], on='image_id')

    ids_qc_df.to_csv(experiment_dir / qc_output_filename, index=False)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file,
         args.mriqc_threshold, args.qoala_threshold)
