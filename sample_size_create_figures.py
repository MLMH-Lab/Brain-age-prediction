#!/usr/bin/env python3
"""Plot results of bootstrap analysis

Ref:
https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-M', '--model_name',
                    dest='model_name',
                    help='Name of the model.')

parser.add_argument('-N', '--n_bootstrap',
                    dest='n_bootstrap',
                    type=int, default=1000,
                    help='Number of bootstrap iterations.')

parser.add_argument('-R', '--n_max_pair',
                    dest='n_max_pair',
                    type=int, default=20,
                    help='Number maximum of pairs.')

args = parser.parse_args()


def main(experiment_name, model_name, n_bootstrap, n_max_pair):
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    i_n_subject_pairs_list = range(1, n_max_pair+1)

    scores_i_n_subject_pairs = []
    train_scores_i_n_subject_pairs = []

    for i_n_subject_pairs in i_n_subject_pairs_list:
        ids_with_n_subject_pairs_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}'
        scores_dir = ids_with_n_subject_pairs_dir / 'scores'
        scores_bootstrap = []
        train_scores_bootstrap = []
        for i_bootstrap in range(n_bootstrap):
            filepath_scores = scores_dir / f'scores_{i_bootstrap:04d}_{model_name}.npy'
            scores_bootstrap.append(np.load(str(filepath_scores))[1])

            train_filepath_scores = scores_dir / f'scores_{i_bootstrap:04d}_{model_name}_train.npy'
            train_scores_bootstrap.append(np.load(str(train_filepath_scores))[1])

        scores_i_n_subject_pairs.append(scores_bootstrap)
        train_scores_i_n_subject_pairs.append(train_scores_bootstrap)

    age_min = 47
    age_max = 73
    std_uniform_dist = np.sqrt(((age_max - age_min) ** 2) / 12)

    plt.figure(figsize=(20, 5))

    # Draw lines
    plt.plot(i_n_subject_pairs_list,
             np.mean(scores_i_n_subject_pairs, axis=1),
             color='#111111', label=model_name+' performance')

    plt.plot(i_n_subject_pairs_list, std_uniform_dist * np.ones_like(i_n_subject_pairs_list), '--',
             color='#111111', label='Chance line')

    plt.plot(i_n_subject_pairs_list,
             np.mean(train_scores_i_n_subject_pairs, axis=1),
             color='#111111', label=model_name+' train performance')

    # Draw bands
    plt.fill_between(i_n_subject_pairs_list,
                     np.percentile(scores_i_n_subject_pairs, 2.5, axis=1),
                     np.percentile(scores_i_n_subject_pairs, 97.5, axis=1),
                     color='#DDDDDD')

    plt.fill_between(i_n_subject_pairs_list,
                     np.percentile(train_scores_i_n_subject_pairs, 2.5, axis=1),
                     np.percentile(train_scores_i_n_subject_pairs, 97.5, axis=1),
                     color='#DDDDDD')

    # Create plot
    plt.title('Bootstrap Analysis')
    plt.xlabel('Number of subjects')
    plt.xticks(i_n_subject_pairs_list, np.multiply(i_n_subject_pairs_list, 2 * ((73 - 47) + 1)))
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(str(experiment_dir / 'sample_size' / f'sample_size_{model_name}.png'))


if __name__ == '__main__':
    main(args.experiment_name, args.model_name,
         args.n_bootstrap, args.n_max_pair)
