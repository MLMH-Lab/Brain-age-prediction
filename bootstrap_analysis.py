"""Plot results of bootstrap analysis"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    # ----------------------------------------------------------------------------------------

    i_n_subject_pairs_list = range(1, 51)
    n_bootstrap = 1000

    scores_i_n_subject_pairs_mean = []
    scores_i_n_subject_pairs_std = []

    for i_n_subject_pairs in i_n_subject_pairs_list:
        ids_with_n_subject_pairs_dir = experiment_dir / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subject_pairs))
        scores_dir = ids_with_n_subject_pairs_dir / 'scores'
        scores_bootstrap = []
        for i_bootstrap in range(n_bootstrap):
            # Save arrays with permutation coefs and scores as np files
            filepath_scores = scores_dir / ('boot_scores_{:04d}.npy'.format(i_bootstrap))
            scores_bootstrap.append(np.load(str(filepath_scores))[1])

        scores_i_n_subject_pairs_mean.append(np.mean(scores_bootstrap))
        scores_i_n_subject_pairs_std.append(np.std(scores_bootstrap))

    age_min = 47
    age_max = 73
    std = np.sqrt(((age_max - age_min) ** 2) / 12)

    fig = plt.figure(figsize=(20, 5))

    # Draw lines
    plt.plot(i_n_subject_pairs_list, scores_i_n_subject_pairs_mean, color="#111111", label="SVM performance")
    plt.plot(i_n_subject_pairs_list, std * np.ones_like(i_n_subject_pairs_list), '--', color="#111111",
             label="Chance line")

    # Draw bands
    plt.fill_between(i_n_subject_pairs_list,
                     scores_i_n_subject_pairs_mean - np.multiply(1.96, scores_i_n_subject_pairs_std),
                     scores_i_n_subject_pairs_mean + np.multiply(1.96, scores_i_n_subject_pairs_std),
                     color="#DDDDDD")

    # Create plot
    plt.title("Bootstrap Analysis")
    plt.xlabel("Number of subjects")
    plt.xticks(i_n_subject_pairs_list, np.multiply(i_n_subject_pairs_list, (73 - 47)))
    plt.ylabel("Mean Absolute Error")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(str(experiment_dir / 'bootstrap_analysis' / 'bootstrap_analysis.png'))


if __name__ == "__main__":
    main()
