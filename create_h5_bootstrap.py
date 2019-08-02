"""Script to create bootstrap datasets of UK BIOBANK Scanner1 in hdf5 format using create_h5_dataset script;
Uses ids from create_bootstrap_ids
"""
from pathlib import Path

from create_h5_dataset import create_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    # ----------------------------------------------------------------------------------------

    # Loop over the 50 bootstrap samples with up to 50 gender-balanced subject pairs per age group/year
    for i_n_subject_pairs in range(1, 51):
        print(i_n_subject_pairs)

        ids_with_n_subject_pairs_dir = experiment_dir / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subject_pairs))

        dataset_dir = ids_with_n_subject_pairs_dir / 'datasets'
        dataset_dir.mkdir(exist_ok=True)

        # Loop over the 1000 random subject samples per bootstrap
        n_bootstrap = 1000
        for i_bootstrap in range(n_bootstrap):
            ids_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}.csv'.format(i_bootstrap, i_n_subject_pairs)
            id_path = ids_with_n_subject_pairs_dir / 'ids' / ids_filename

            dataset_filename = 'homogeneous_bootstrap_{:04d}_n_{:02d}.h5'.format(i_bootstrap, i_n_subject_pairs)
            dataset_path = dataset_dir / dataset_filename

            create_dataset(demographic_path, id_path, freesurfer_path, dataset_path)


if __name__ == "__main__":
    main()
