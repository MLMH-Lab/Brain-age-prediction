#!/usr/bin/env python3
"""Script to create the Kernel matrix (Gram matrix).

The Kernel matrix will be used on the analysis with voxel data.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-P', '--input_path',
                    dest='input_path_str',
                    help='Path to the local folder with preprocessed images.')

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

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


def calculate_gram_matrix(subjects_path, mask_img, step_size=1000):
    """Calculate the Gram matrix.

    Args:
        subjects_path:
        mask_img:
        step_size:

    Returns:

    """
    n_samples = len(subjects_path)
    gram_matrix = np.float64(np.zeros((n_samples, n_samples)))

    # Outer loop
    outer_pbar = tqdm(range(int(np.ceil(n_samples / np.float(step_size)))))
    for ii in outer_pbar:
        outer_pbar.set_description(f'Processing outer loop {ii}')
        # Generate indices and then paths for this block
        start_ind_1 = ii * step_size
        stop_ind_1 = min(start_ind_1 + step_size, n_samples)
        block_paths_1 = subjects_path[start_ind_1:stop_ind_1]

        # Read in the images in this block
        images_1 = []
        images_1_pbar = tqdm(block_paths_1)
        for path in images_1_pbar:
            images_1_pbar.set_description(f'Loading outer image {path}')
            try:
                img = nib.load(str(path))
            except FileNotFoundError:
                print(f'No image file {path}.')
                raise

            # Extract only the brain voxels. This will create a 1D array.
            img = apply_mask(img, mask_img)
            img = np.asarray(img, dtype='float64')
            img = np.nan_to_num(img)
            images_1.append(img)
            del img
        images_1 = np.array(images_1)

        # Inner loop
        inner_pbar = tqdm(range(ii + 1))
        for jj in inner_pbar:

            # If ii = jj, then sets of image data are the same - no need to load
            if ii == jj:
                start_ind_2 = start_ind_1
                stop_ind_2 = stop_ind_1
                images_2 = images_1

            # If ii !=jj, read in a different block of images
            else:
                # Generate indices and then paths for this block
                start_ind_2 = jj * step_size
                stop_ind_2 = min(start_ind_2 + step_size, n_samples)
                block_paths_2 = subjects_path[start_ind_2:stop_ind_2]

                images_2 = []
                images_2_pbar = tqdm(block_paths_2)
                for path in images_2_pbar:
                    images_2_pbar.set_description(f'Loading inner image {path}')
                    try:
                        img = nib.load(str(path))
                    except FileNotFoundError:
                        print(f'No image file {path}.')
                        raise

                    img = apply_mask(img, mask_img)
                    img = np.asarray(img, dtype='float64')
                    img = np.nan_to_num(img)
                    images_2.append(img)
                    del img
                images_2 = np.array(images_2)

            block_K = np.dot(images_1, np.transpose(images_2))
            gram_matrix[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
            gram_matrix[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

    return gram_matrix


def main(input_path_str, experiment_name, input_ids_file, input_data_type, mask_filename):
    """"""
    dataset_path = Path(input_path_str)

    output_path = PROJECT_ROOT / 'outputs' / 'kernels'
    output_path.mkdir(exist_ok=True, parents=True)

    ids_df = pd.read_csv(PROJECT_ROOT / 'outputs' / experiment_name / input_ids_file)

    # Get list of subjects included in the analysis
    subjects_path = [str(dataset_path / f'{subject_id}_Warped{input_data_type}') for subject_id in ids_df['image_id']]

    print(f'Total number of images: {len(ids_df)}')

    # ----------------------------------------------------------------------------------------
    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
    mask_img = nib.load(str(brain_mask))

    gram_matrix = calculate_gram_matrix(subjects_path, mask_img)

    gram_df = pd.DataFrame(columns=ids_df['image_id'].tolist(), data=gram_matrix)
    gram_df['image_id'] = ids_df['image_id']
    gram_df = gram_df.set_index('image_id')

    gram_df.to_csv(output_path / 'kernel.csv')


if __name__ == '__main__':
    main(args.input_path_str, args.experiment_name,
         args.input_ids_file,
         args.input_data_type, args.mask_filename)
