#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="MIDAS TCN Benchmark"
#SBATCH --error="./logs/job-%j-tcn_train_script.err"
#SBATCH --output="./logs/job-%j-tcn_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

CFG_PATH="./configs/TCN/exp_trakstar_sw_handkp_10hz.yaml"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&
# python -u train_recognition.py --job_id 5555555  &&
python -u TCN_train_recognition.py --job_id "$SLURM_JOB_ID" --config "$CFG_PATH" &&
# python -u TCN_train_recognition.py --job_id 0 --config "$CFG_PATH" &&
echo "Done" &&
exit
