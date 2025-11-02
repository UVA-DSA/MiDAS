#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Bootcamp Pedal Extract Benchmark"
#SBATCH --error="./logs/job-%j-pedal_extraction.err"
#SBATCH --output="./logs/job-%j-pedal_extraction.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda activate egoexoems &&
# python -u extract_camera_events.py --job_id "$SLURM_JOB_ID"  && # camera pedal -- good hamid
# python -u extract_camera_pedal.py --job_id "$SLURM_JOB_ID"  && # not so good keshara
# python -u extract_arm_swap_pedal.py --job_id "$SLURM_JOB_ID"  && # armswap -- good hamid
python -u extract_pri_sec_pedals.py --job_id "$SLURM_JOB_ID"  && # yellow and blue pedals -- good hamid


echo "Done" &&
exit
