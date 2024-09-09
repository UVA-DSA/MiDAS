#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Synch RAS Bootcamp Data"
#SBATCH --error="./logs/job-%j-synch_ras_data_script.err"
#SBATCH --output="./logs/job-%j-synch_ras_data_script.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate cogems &&
python -u main.py && 
echo "Done" &&
exit
