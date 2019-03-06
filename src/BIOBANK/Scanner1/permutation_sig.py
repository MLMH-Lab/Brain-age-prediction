"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')

def main():
    # Define what subjects were modeled: total, male or female
    subjects = 'total'

    # Load permutation coefficients
    perm_coef = np.load(PROJECT_ROOT / 'outputs' / 'permutations' / subjects / 'perm_coef.npy')

    # Load permutation scores
    perm_scores = np.load(PROJECT_ROOT / 'outputs' / 'permutations' / subjects / 'perm_scores.npy')


if __name__ == "__main__":
    main()