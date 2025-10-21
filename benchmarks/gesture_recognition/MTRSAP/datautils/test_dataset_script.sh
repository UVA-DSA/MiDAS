#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="MIDAS Dataset Test"
#SBATCH --error="./logs/job-%j-midas_dataset_test.err"
#SBATCH --output="./logs/job-%j-midas_dataset_test.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&
# python -u train_recognition.py --job_id 5555555  &&
python -u midas.py --job_id "$SLURM_JOB_ID"  &&
echo "Done" &&
exit
