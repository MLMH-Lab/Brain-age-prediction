#!/usr/bin/env python3
""" Script to calculate the primal weights of the SVM approach
for voxel data.

"""
import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import load
from nilearn.masking import apply_mask
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-P', '--input_path',
                    dest='input_path_str',
                    help='Path to the local folder with preprocessed images.')

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='homogenized_ids.csv',
                    help='Filename indicating the ids to be used.')

parser.add_argument('-D', '--input_data_type',
                    dest='input_data_type',
                    default='.nii.gz',
                    help='Input data type')

parser.add_argument('-M', '--mask_filename',
                    dest='mask_filename',
                    default='mni_icbm152_t1_tal_nlin_sym_09c_mask.nii',
                    help='Input data type')

args = parser.parse_args()


def main(experiment_name, input_path_str, input_ids_file, input_data_type, mask_filename):
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    dataset_path = Path(input_path_str)

    model_dir = experiment_dir / 'voxel_SVM'
    cv_dir = model_dir / 'cv'

    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / input_ids_file
    ids_df = pd.read_csv(ids_path)

    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
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
            prefix = f'{i_repetition:02d}_{i_fold:02d}'
            model = load(cv_dir / f'{prefix}_regressor.joblib')

            # Load train index
            train_index = np.load(cv_dir / f'{prefix}_train_index.npy')

            coef_list.append(model.dual_coef_[0])
            index_list.append(train_index[model.support_])

    # number of voxels in the mask
    mask_data = mask_img.get_fdata()
    n_voxels = sum(sum(sum(mask_data > 0)))
    n_models = 100
    weights = np.zeros((n_models, n_voxels))

    for i, subject_id in enumerate(tqdm(ids_df['Image_ID'])):
        # Check if subject is support vector in any model before load the image.
        is_support_vector = False
        for support_index in index_list:
            if i in support_index:
                is_support_vector = True
                break

        if is_support_vector == False:
            continue

        subject_id = subject_id.rstrip('/')
        subject_path = dataset_path / f'{subject_id}_Warped{input_data_type}'

        try:
            img = nib.load(str(subject_path))
        except FileNotFoundError:
            print(f'No image file {subject_path}.')
            raise

        # Extract only the brain voxels. This will create a 1D array.
        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float64')
        img = np.nan_to_num(img)

        for j, (dual_coef, support_index) in enumerate(zip(coef_list, index_list)):
            if i in support_index:
                selected_dual_coef = dual_coef[np.argwhere(support_index == i)]
                weights[j, :] = weights[j, :] + selected_dual_coef * img

    coords = np.argwhere(mask_data > 0)
    i = 0
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            importance_map = np.zeros_like(mask_data)
            for xyz, importance in zip(coords, weights[i, :]):
                importance_map[tuple(xyz)] = importance

            importance_map_nifti = nib.Nifti1Image(importance_map, np.eye(4)) #TODO: use mask affine
            importance_filename = f'{i_repetition:02d}_{i_fold:02d}_importance.nii.gz'
            nib.save(importance_map_nifti, str(cv_dir / importance_filename))
            i = i + 1


if __name__ == '__main__':
    main(args.experiment_name, args.input_path_str,
         args.input_ids_file,
         args.input_data_type, args.mask_filename)
