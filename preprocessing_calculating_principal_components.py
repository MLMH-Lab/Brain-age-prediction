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
import warnings

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
                     ids_df['Image_ID'].str.rstrip('/')]

    print(f'Total number of images: {len(ids_df)}')

    # ----------------------------------------------------------------------------------------
    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
    mask_img = nib.load(str(brain_mask))

    n_repetitions = 10
    n_folds = 10
    prefix_list = []
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            prefix = f'{i_repetition:02d}_{i_fold:02d}'
            prefix_list.append(prefix)

    n_models = 100
    step = 10

    component_list = []
    for _ in range(100):
        component_list.append(np.zeros((len(subjects_path), 200)))

    for i_step in range(n_models//step):
        print(i_step)
        print(i_step*step)
        print((i_step+1) * step)

        models = []
        for i_model in range(i_step*step, (i_step+1) * step):
            model = load(models_output_path / f'{prefix_list[i_model]}_pca.joblib')
            models.append(model)
            del model

        for i_subj, subject_path in enumerate(subjects_path):
            print(subject_path)
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

            for model, i_model in zip(models, range(i_step*step, (i_step+1) * step)):
                transformed = model.transform(img[None, :])
                component_list[i_model][i_subj, :] = transformed

    for i_comp, component in enumerate(component_list):
        pca_df = pd.DataFrame(data = component)
        pca_df['Image_ID'] = subjects_path
        pca_df.to_csv(output_path / f'{prefix_list[i_comp]}_pca_components.csv', index=False)


if __name__ == '__main__':
    main(args.input_path_str, args.experiment_name,
         args.input_ids_file, args.scanner_name,
         args.input_data_type, args.mask_filename)
