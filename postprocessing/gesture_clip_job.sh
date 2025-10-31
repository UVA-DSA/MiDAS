#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Bootcamp Gesture Clip Creation"
#SBATCH --error="./logs/job-%j-gesture_clip_creation_script.err"
#SBATCH --output="./logs/job-%j-gesture_clip_creation_script.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"


module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda activate egoexoems &&
# python -u train_recognition.py --job_id 5555555  &&
# python -u MTRSAP_train_recognition.py --job_id "$SLURM_JOB_ID" --config "$CFG_PATH" &&
python -u gesture_clip_creator.py  &&
echo "Done" &&
exit
