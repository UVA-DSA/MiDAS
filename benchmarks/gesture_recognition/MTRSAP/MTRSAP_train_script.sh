#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="MIDAS MTRSAP Benchmark"
#SBATCH --error="./logs/job-%j-mtrsap_train_script.err"
#SBATCH --output="./logs/job-%j-mtrsap_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

CFG_PATH="./davinci_configs/MTRSAP/exp_image_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_10hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_image_30hz.yaml"

# CFG_PATH="./desk_configs/MTRSAP/exp_image_10hz.yaml"
# CFG_PATH="./desk_configs/MTRSAP/exp_psm_10hz.yaml"
# CFG_PATH="./jigsaws_configs/MTRSAP/exp_psm_10hz.yaml"
# CFG_PATH="./desk_configs/MTRSAP/exp_image_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_image_10hz.yaml"

CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_image_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_image_vit_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_image_vit_30hz.yaml"
CFG_PATH="./davinci_configs/MTRSAP/exp_handkp_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_handkp_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_image_handkp_30hz.yaml"
# CFG_PATH="./davinci_configs/MTRSAP/exp_trakstar_image_vit_handkp_30hz.yaml"



# CFG_PATH="./desk_configs/MTRSAP/exp_image_10hz.yaml"
# CFG_PATH="./desk_configs/MTRSAP/exp_image_vit_10hz.yaml"
# CFG_PATH="./desk_configs/MTRSAP/exp_psm_10hz.yaml"
# CFG_PATH="./desk_configs/MTRSAP/exp_psm_image_10hz.yaml"
# CFG_PATH="./desk_configs/MTRSAP/exp_psm_image_vit_10hz.yaml"

# CFG_PATH="./jigsaws_configs/MTRSAP/exp_image_30hz.yaml"
# CFG_PATH="./jigsaws_configs/MTRSAP/exp_psm_30hz.yaml"
# CFG_PATH="./jigsaws_configs/MTRSAP/exp_psm_image_30hz.yaml"

# CFG_PATH="./raven_configs/MTRSAP/exp_raven_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_image_vit_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_trakstar_sw_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_trakstar_sw_image_vit_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_image_resnet_raven_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_trakstar_10hz.yaml"
# CFG_PATH="./raven_configs/MTRSAP/exp_trakstar_sw_image_resnet_10hz.yaml"


module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda activate egoexoems &&
# python -u train_recognition.py --job_id 5555555  &&
python -u MTRSAP_train_recognition.py --job_id "$SLURM_JOB_ID" --config "$CFG_PATH" --wandb on &&
# python -u MTRSAP_train_recognition.py --job_id 0 --config "$CFG_PATH" --wandb off &&
echo "Done" &&
exit
