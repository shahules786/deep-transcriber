#!/bin/bash
set -e

echo "Loading Anaconda Module"
module load anaconda

echo "Creating Virtual Environment"
conda env create -f environment.yml ||  conda env update -f environment.yml

echo "Fetch Bert Model..."
source activate pytorch-ner
python src/config.py