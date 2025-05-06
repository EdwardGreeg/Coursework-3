#!/bin/bash
#PBS -N modeling_job
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=24:00:00
#PBS -l mem=64gb
#PBS -o $PBS_O_WORKDIR/output.log
#PBS -e $PBS_O_WORKDIR/error.log
#PBS -V

cd $PBS_O_WORKDIR

module load ScriptLang/python/3.8.3
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH

pip install --user git+https://github.com/open2c/polychrom.git

pip install --user numpy h5py openmm matplotlib seaborn pandas 

if [ $# -ne 1 ]; then
  echo "Usage: $0 <NUM_BLOCKS>"
  exit 1
fi

NUM_BLOCKS=$1

PYTHON_INTERPRETER="python3"

read -r -d '' PYTHON_FUNCTIONS << EOPY
import numpy as np
import os
import json
from collections import Counter
from my_module import generate_counters_and_interactions, generate_random_ensemble_from_probabilistic_counters

letters = 'ABCDEFGHIJKLMNOP'
di = {index: letter for index, letter in enumerate(letters, start=1)}

def save_ensemble(ensemble, path):
    os.makedirs(path, exist_ok=True)
    for i, row in enumerate(ensemble):
      row = np.array([di[c] for c in row])
      filename = os.path.join(path, f"random_compartment{i}.json")
      with open(filename, "w") as f:
          json.dump(row.tolist(), f)
EOPY


mkdir -p results zips ensembles

# Основной цикл
for ((i=0; i<NUM_BLOCKS; i++)); do
  echo "=== Запуск блока $i ==="

  # Подготовка путей
  MARKUP_PATH="./inputs"
  ZIP_FILE="zips/block$i.zip"
  RESULT_DIR="results/block$i"
  ENSEMBLE_FILE="ensembles/ensemble$i.npy"

  mkdir -p "$MARKUP_PATH"

  # Генерация структуры
  $PYTHON_INTERPRETER -c "$PYTHON_FUNCTIONS
counters, interaction_matrix = generate_counters_and_interactions(segment_min=20, dominant_fraction_range = (0.3, 0.8), segment_max=100, min_coverage = 0.70,
                                               type1_spike_prob=0.3, type1_high_range=(1000, 4500), type1_low_range=(0, 1000),
                                               dominant_type_cap=4000, subsegment_variation=0.2, localization_penalty=0.1)
                                               
ensemble = generate_random_ensemble_from_probabilistic_counters(counters, n_structures=100, beads_per_bin=15, seed=None)
np.save('$ENSEMBLE_FILE', ensemble)
np.save('./inputs/interaction_matrix.npy', interaction_matrix)
save_ensemble(ensemble, '$MARKUP_PATH')
"

  # Запуск симуляции
  $PYTHON_INTERPRETER 01_simulate.py \
    --compartment-markup "$MARKUP_PATH" \
    --result-dir "$RESULT_DIR" \
    --n-chromosomes 1 \
    --num-runs 100 \
    --num-blocks 100 \
    --block-steps 400 \
    --save-starting-conformation \
    --save-block-conformation

  # Архивация PDB-файлов

  zip -j "zips/block$i.zip" results/block$i/*.pdb
done
