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
from joblib import load
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

parser.add_argument('-S', '--output_suffix',
                    dest='output_suffix',
                    default='',
                    help='Filename indicating the ids to be used.')

args = parser.parse_args()

def load_all_subjects(subjects_path, mask_img):
    imgs = []
    subj_pbar = tqdm(subjects_path)
    for subject_path in subj_pbar:
        subj_pbar.set_description(f'Loading outer image {subject_path}')
        # Read in the images in this block
        try:
            img = nib.load(str(subject_path))
        except FileNotFoundError:
            print(f'No image file {subject_path}.')
            raise

        # Extract only the brain voxels. This will create a 1D array.
        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        imgs.append(img)
    return imgs

def main(input_path_str, experiment_name, input_ids_file, input_data_type, mask_filename, output_suffix):
    """"""
    dataset_path = Path(input_path_str)
    output_path = PROJECT_ROOT / 'outputs' / 'pca'
    models_output_path = output_path / 'models'

    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / input_ids_file
    ids_df = pd.read_csv(ids_path)

    # Get list of subjects included in the analysis
    subjects_path = [str(dataset_path / f'{subject_id}_Warped{input_data_type}') for subject_id in ids_df['image_id']]

    print(f'Total number of images: {len(ids_df)}')

    # ----------------------------------------------------------------------------------------
    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
    mask_img = nib.load(str(brain_mask))

    imgs = load_all_subjects(subjects_path, mask_img)

    n_components = 150
    n_repetitions = 10
    n_folds = 10
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            prefix = f'{i_repetition:02d}_{i_fold:02d}'
            print(f'{prefix}')

            components = np.zeros((len(subjects_path), n_components))
            model = load(models_output_path / f'{prefix}_pca.joblib')

            for i_img, img in enumerate(tqdm(imgs)):
                components[i_img, :] = model.transform(img[None, :])

            pca_df = pd.DataFrame(data=components)
            pca_df['image_id'] = subjects_path
            pca_df.to_csv(output_path / f'{prefix}_pca_components{output_suffix}.csv', index=False)


if __name__ == '__main__':
    main(args.input_path_str, args.experiment_name,
         args.input_ids_file,
         args.input_data_type, args.mask_filename, args.output_suffix)
