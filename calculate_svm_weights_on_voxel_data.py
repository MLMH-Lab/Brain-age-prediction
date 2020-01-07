""" Script to calculate the primal weights of the SVM approach
for voxel data.

"""
from pathlib import Path
import random

import numpy as np
import nibabel as nib
from sklearn.externals.joblib import load
from nilearn.masking import apply_mask
import pandas as pd
from tqdm import tqdm
import argparse
import sys

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser("Type of Vector Machine.")
parser.add_argument('vm', dest='vm_type', nargs='?', default='svm')

def main(vm_type):
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_path = Path('/media/kcl_1/SSD2/BIOBANK/')
    input_data_type = '.nii.gz'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    if vm_type == 'svm':
        vm_dir = experiment_dir / 'voxel_SVM'
    if vm_type == 'rvm':
        vm_dir = experiment_dir / 'voxel_RVM'
    cv_dir = vm_dir / 'cv'

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
    coef_list = []
    index_list = []
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            # Load model
            model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
            vm = load(cv_dir / model_filename)

            # Load train index
            index_filename = '{:02d}_{:02d}_train_index.npy'.format(i_repetition, i_fold)
            train_index = np.load(cv_dir / index_filename)

            if vm_type=='svm':
                coef_list.append(vm.dual_coef_[0])
                index_list.append(train_index[vm.support_])
            else: #rvm
                coef_list.append(vm.mu_)
                index_list.append(train_index[vm.relevance_])

    # number of voxels in the mask
    n_voxels = 1886539
    weights = np.zeros((100, n_voxels))

    for i, subject_id in enumerate(tqdm(subject_ids['Participant_ID'])):
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

        for j, (dual_coef, support_index) in enumerate(zip(coef_list, index_list)):
            if i in support_index:
                selected_dual_coef = dual_coef[np.argwhere(support_index == i)]
                weights[j, :] = weights[j, :] + selected_dual_coef * img

    mask_data = mask_img.get_fdata()
    coords = np.argwhere(mask_data > 0)
    i = 0
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            importance_map = np.zeros_like(mask_data)
            for xyz, importance in zip(coords, weights[i, :]):
                importance_map[tuple(xyz)] = importance

            importance_map_nifti = nib.Nifti1Image(importance_map, np.eye(4))
            importance_filename = '{:02d}_{:02d}_importance.nii.gz'.format(i_repetition, i_fold)
            nib.save(importance_map_nifti, str(cv_dir / importance_filename))
            i = i + 1


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    main(vm_type=args.vm_type)
