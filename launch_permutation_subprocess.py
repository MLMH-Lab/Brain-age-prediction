#!/usr/bin/env python3
from pathlib import Path
import subprocess

PROJECT_ROOT = Path.cwd()
venv_python = PROJECT_ROOT / 'venv' / 'bin' / 'python3'
code_perm = PROJECT_ROOT / 'permutation.py'

n_threads = 6
initial_index_perm = 0
number_perm_per_thread = 25

slices_indexes = []
for i in range(n_threads):
    slices_indexes.append([initial_index_perm + i*number_perm_per_thread,  initial_index_perm+ (i+1)*number_perm_per_thread])

processes = []

for slices_i in slices_indexes:
    command = str(venv_python) + ' ' + str(code_perm) + ' %d %d'%(slices_i[0], slices_i[1])
    print(command)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    processes.append(p)

for p in processes:
    p.wait()
