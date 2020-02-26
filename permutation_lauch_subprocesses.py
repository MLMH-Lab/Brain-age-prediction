#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

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

args = parser.parse_args()

PROJECT_ROOT = Path.cwd()
venv_python = PROJECT_ROOT / 'venv' / 'bin' / 'python3'
code_perm = PROJECT_ROOT / 'src' / 'permutation' / 'permutation_train_svm_models.py'

n_threads = 2
initial_index_perm = 0
number_perm_per_thread = 25

slices_indexes = []
for i in range(n_threads):
    slices_indexes.append([initial_index_perm + i * number_perm_per_thread,
                           initial_index_perm + (
                                       i + 1) * number_perm_per_thread])

processes = []

for slices_i in slices_indexes:
    command = (f'{str(venv_python)} {str(code_perm)} -E {args.experiment_name}'
               f' -S {args.scanner_name} -K {slices_i[0]} -L {slices_i[1]}')
    print(command)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    processes.append(p)

for p in processes:
    p.wait()
