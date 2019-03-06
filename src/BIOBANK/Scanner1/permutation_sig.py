"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np

import os
import glob

from sklearn.externals import joblib

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')

def main():
    # Define what subjects were modeled: total, male or female
    subjects = 'total'

    # Load permutation coefficients
    perm_coef = np.load(PROJECT_ROOT / 'outputs' / 'permutations' / subjects / 'perm_coef.npy')

    # Get SVM model file names
    model_file_names = []
    output_dir = str(PROJECT_ROOT / 'outputs' / subjects)
    for file_path in glob.iglob(output_dir + '*/*svm.joblib', recursive=True):
        file_name = os.path.basename(file_path)
        model_file_names.append(file_name)

    # Get SVM model coefficients
    n_models = len(model_file_names)
    model_array_coef = np.zeros([n_models, 100])
    index = 0
    for model_name in model_file_names:
        model = joblib.load(PROJECT_ROOT / 'outputs' / subjects / model_name)
        model_coef = model.coef_
        model_array_coef[index] = model_coef
        index += 1

    # Get mean SVM model coefficients
    model_array_coef_abs = np.abs(model_array_coef)
    model_coef_mean = np.mean(model_array_coef_abs, axis=0)

    # Get proportion of permuted coefficients higher than actual coefficients

    # Create array of permuted coef subtracted by actual coef
    diff_array = np.zeros([n_models, 100])

    ind = 0
    for row in model_array_coef:
        new_row = row - model_coef_mean
        diff_array[ind] = new_row
        ind += 1

    # # per feature, count how often negative + divide by number of permutations
    # n_perm = len(perm_coef)
    #
    # ind2 = 0
    # for f in diff_array:
    #     if diff_array[ind2] < 0:
    #         count ...



if __name__ == "__main__":
    main()