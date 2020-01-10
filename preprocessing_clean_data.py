#!/usr/bin/env python3
"""Clean UK Biobank data.

Most of the subjects are white and some ages have very low number of subjects (<100).
The ethnics minorities and age with low number are removed from further analysis
as well subjects with any mental or brain disorder.
"""
import argparse
from pathlib import Path

from utils import load_demographic_data

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
                    default='freesurferData.csv',
                    help='Filename indicating the ids to be used.')
args = parser.parse_args()


def main(experiment_name, scanner_name, input_ids_file):
    """Clean UK Biobank data."""
    # ----------------------------------------------------------------------------------------
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / input_ids_file

    output_ids_filename = 'cleaned_ids_noqc.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(participants_path, ids_path)

    # Exclude subjects outside [47, 73] interval (ages with <100 participants).
    dataset = dataset.loc[(dataset['Age'] >= 47) & (dataset['Age'] <= 73)]

    # Exclude non-white ethnicities due to small subgroups
    dataset = dataset[dataset['Ethnicity'] == 'White']

    # Exclude patients
    dataset = dataset[dataset['Diagn'] == 1]

    output_ids_df = dataset[['Image_ID']]

    assert sum(output_ids_df.duplicated(subset='Image_ID')) == 0

    output_ids_df.to_csv(experiment_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
