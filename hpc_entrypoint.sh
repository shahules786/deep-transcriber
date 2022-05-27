#!/bin/bash
set -e

echo '----------------------------------------------------'
echo ' SLURM_CLUSTER_NAME = '$SLURM_CLUSTER_NAME
echo '    SLURMD_NODENAME = '$SLURMD_NODENAME
echo '        SLURM_JOBID = '$SLURM_JOBID
echo '     SLURM_JOB_USER = '$SLURM_JOB_USER
echo '    SLURM_PARTITION = '$SLURM_JOB_PARTITION
echo '  SLURM_JOB_ACCOUNT = '$SLURM_JOB_ACCOUNT
echo '----------------------------------------------------'

#TeamCity Output
cat << EOF
##teamcity[buildNumber '$SLURM_JOBID']
EOF

echo "Load HPC modules"
module load anaconda

echo "Activate Environment"
source activate deep-transcriber
export TRANSFORMERS_OFFLINE=True
export PYTHONPATH=${PYTHONPATH}:$/scratch/$USER/deep-transcriber

source ~/mlflow_settings.sh

echo "Making temp dir"
mkdir temp

echo "Start Training..."
python transcriber/tasks/embeddings/trainer.py
