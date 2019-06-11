"""Script to create bootstrap datasets of UK BIOBANK Scanner1
in hdf5 format using create_h5_dataset script"""
from pathlib import Path

from create_h5_dataset import create_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    i_n_subjects = 1
    ids_with_n_subjects_dir = PROJECT_ROOT / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subjects))

    n_bootstrap = 1000
    # ----------------------------------------------------------------------------------------
    # Loop over the 50 samples created in script create_bootstrap_ids
    for i_bootstrap in range(n_bootstrap):
        ids_filename = 'homogeneous_bootstrap_{:02d}_n_{:02d}.csv'.format(i_bootstrap, i_n_subjects)

        file_name = os.path.basename(file_path)
        print(file_name)
        create_dataset(dataset_homogeneous=file_name,
                       input_dir='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/bootstrap_ids',
                       output_dir='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/h5_datasets')


if __name__ == "__main__":
    main()
