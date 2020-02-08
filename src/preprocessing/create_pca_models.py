#!/usr/bin/env python3
"""Script to create the Kernel matrix (Gram matrix).

The Kernel matrix will be used on the analysis with voxel data.
"""
import argparse
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import dump
from nilearn.masking import apply_mask
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-P', '--input_path',
                    dest='input_path_str',
                    help='Path to the local folder with preprocessed images.')

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

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


def main(input_path_str, experiment_name, input_ids_file, scanner_name, input_data_type, mask_filename):
    """"""
    dataset_path = Path(input_path_str)

    output_path = PROJECT_ROOT / 'outputs' / 'pca'
    output_path.mkdir(exist_ok=True)

    models_output_path = output_path / 'models'
    models_output_path.mkdir(exist_ok=True)

    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / input_ids_file
    ids_df = pd.read_csv(ids_path)

    # Get list of subjects included in the analysis
    subjects_path = [str(dataset_path / f'{subject_id}_Warped{input_data_type}') for subject_id in
                     ids_df['image_id'].str.rstrip('/')]

    print(f'Total number of images: {len(ids_df)}')

    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'

    dataset = load_demographic_data(participants_path, ids_path)

    age = dataset['Age'].values

    # ----------------------------------------------------------------------------------------
    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
    mask_img = nib.load(str(brain_mask))

    n_repetitions = 10
    n_folds = 10
    step_size = 850
    for i_repetition in range(n_repetitions):
        # Create 10-fold cross-validation scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(age, age)):
            print(f'Running repetition {i_repetition:02d}, fold {i_fold:02d}')
            print(train_index.shape)
            n_samples = len(subjects_path)
            pca = IncrementalPCA(n_components=800, copy=False)

            for i in tqdm(range(int(np.ceil(n_samples / np.float(step_size))))):
                # Generate indices and then paths for this block
                start_ind = i * step_size
                stop_ind = min(start_ind + step_size, n_samples)
                block_paths = subjects_path[start_ind:stop_ind]

                # Read in the images in this block
                images = []
                for path in tqdm(block_paths):
                    try:
                        img = nib.load(str(path))
                    except FileNotFoundError:
                        print(f'No image file {path}.')
                        raise

                    # Extract only the brain voxels. This will create a 1D array.
                    img = apply_mask(img, mask_img)
                    img = np.asarray(img, dtype='float32')
                    img = np.nan_to_num(img)
                    images.append(img)
                    del img
                images = np.array(images, dtype='float32')

                selected_index = train_index[(train_index >= start_ind) & (train_index < stop_ind)] - start_ind
                images_selected = images[selected_index]
                try:
                    pca.partial_fit(images_selected)
                except ValueError:
                    warnings.warn('n_components higher than number of subjects.')

            prefix = f'{i_repetition:02d}_{i_fold:02d}'
            dump(pca, models_output_path / f'{prefix}_pca.joblib')


if __name__ == '__main__':
    main(args.input_path_str, args.experiment_name,
         args.input_ids_file, args.scanner_name,
         args.input_data_type, args.mask_filename)
