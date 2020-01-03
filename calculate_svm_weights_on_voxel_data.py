""" Script to calculate the primal weights of the SVM approach
for voxel data.

"""
from pathlib import Path
import random
import warnings

import numpy as np
import nibabel as nib
from sklearn.externals.joblib import load
from nilearn.masking import apply_mask
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_path = Path('/media/kcl_1/SSD2/BIOBANK/')
    input_data_type = '.nii.gz'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / 'voxel_SVM'
    cv_dir = svm_dir / 'cv'

    freesurfer_ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'dataset_homogeneous.csv'

    subject_ids = pd.read_csv(freesurfer_ids_path)

    # Get list of subjects for which we have data
    subjects_path = [str(dataset_path / '{}_ses-bl_T1w_Warped{}'.format(subject_id, input_data_type)) for subject_id in
                     subject_ids['Participant_ID']]

    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
    mask_img = nib.load(str(brain_mask))

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    n_repetitions = 10
    n_folds = 10
    svm_list = []
    train_index_list = []
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            # Load model
            model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            svm = load(cv_dir / model_filename)
            svm_list.append(svm)

            # Load train index
            index_filename = '{:02d}_{:02d}_train_index.npy'.format(i_repetition, i_fold)
            train_index = np.load(cv_dir / index_filename)
            train_index_list.append(train_index)

    for i, subject_id in tqdm(enumerate(subject_ids['Participant_ID'])):
        # print(i)
        path = str(dataset_path / '{}_ses-bl_T1w_Warped{}'.format(subject_id, input_data_type))

        try:
            img = nib.load(str(path))
        except FileNotFoundError:
            print('No image file {}.'.format(path))
            raise

        # Extract only the brain voxels. This will create a 1D array.
        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float64')
        img = np.nan_to_num(img)

if __name__ == "__main__":
    main()