"""Script to create bootstrap datasets of UK BIOBANK Scanner1
in hdf5 format using create_h5_dataset script"""
from pathlib import Path

from create_h5_dataset import create_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    i_n_subjects = 1
    n_bootstrap = 1000

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    ids_with_n_subjects_dir = PROJECT_ROOT / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subjects))
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'
    # ----------------------------------------------------------------------------------------
    bootstrap_dataset_dir = ids_with_n_subjects_dir / 'datasets'
    # Loop over the 50 samples created in script create_bootstrap_ids
    for i_bootstrap in range(n_bootstrap):
        ids_filename = 'homogeneous_bootstrap_{:02d}_n_{:02d}.csv'.format(i_bootstrap, i_n_subjects)
        create_dataset(demographic_path, id_path, freesurfer_path, bootstrap_dataset_dir)


if __name__ == "__main__":
    main()
