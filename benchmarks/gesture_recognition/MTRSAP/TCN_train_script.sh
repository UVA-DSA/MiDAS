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


# CFG_PATH="./davinci_configs/TCN/exp_image_30hz.yaml"
# CFG_PATH="./davinci_configs/TCN/exp_trakstar_30hz.yaml"
# CFG_PATH="./davinci_configs/TCN/exp_trakstar_image_30hz.yaml"
# CFG_PATH="./davinci_configs/TCN/exp_image_vit_30hz.yaml"
# CFG_PATH="./davinci_configs/TCN/exp_trakstar_image_vit_30hz.yaml"

# CFG_PATH="./desk_configs/TCN/exp_image_10hz.yaml"
# CFG_PATH="./desk_configs/TCN/exp_image_vit_10hz.yaml"
# CFG_PATH="./desk_configs/TCN/exp_psm_10hz.yaml"
# CFG_PATH="./desk_configs/TCN/exp_psm_image_10hz.yaml"
# CFG_PATH="./desk_configs/TCN/exp_psm_image_vit_10hz.yaml"

CFG_PATH="./jigsaws_configs/TCN/exp_image_30hz.yaml"
CFG_PATH="./jigsaws_configs/TCN/exp_psm_30hz.yaml"
CFG_PATH="./jigsaws_configs/TCN/exp_psm_image_30hz.yaml"

# # # CFG_PATH="./raven_configs/TCN/exp_image_vit_10hz.yaml"
# # CFG_PATH="./raven_configs/TCN/exp_trakstar_sw_image_vit_10hz.yaml"
# CFG_PATH="./raven_configs/TCN/exp_image_10hz.yaml"
# CFG_PATH="./raven_configs/TCN/exp_trakstar_sw_image_resnet_10hz.yaml"
# CFG_PATH="./raven_configs/TCN/exp_trakstar_10hz.yaml"

module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&
# python -u train_recognition.py --job_id 5555555  &&
python -u TCN_train_recognition.py --job_id "$SLURM_JOB_ID" --config "$CFG_PATH" --wandb on &&
# python -u TCN_train_recognition.py --job_id 0 --config "$CFG_PATH" &&
echo "Done" &&
exit
