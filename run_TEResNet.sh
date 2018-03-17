#!/bin/bash
#SBATCH --job-name=TEResNet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --output=TEResNet-%j.out
#SBATCH --error=TEResNet-%j.err
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

cd /home/${USER}/Codes/TEGAN

# Setup the environment
module purge
module load cuda/9.0 cudnn/7.0.3
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
source ~/tensorflow-gpu/bin/activate

TASK="TEResNet"
LEARNING_RATE=0.0001
RESBLOCKS=12
ENS=0.2
CON=0.5
PHY=0.0
SUFFIX="_RB${RESBLOCKS}_LR${LEARNING_RATE}_ENS${ENS}_CON${CON}_PHY${PHY}"

CMD="python main.py --task ${TASK} --num_resblock ${RESBLOCKS} --learning_rate ${LEARNING_RATE} --lambda_ens ${ENS} --lambda_con ${CON} --lambda_phy ${PHY} --train_dir /farmshare/user_data/${USER}/TEGAN/Data/train --dev_dir /farmshare/user_data/${USER}/TEGAN/Data/dev --max_iter 10000 --max_epoch 50 --save_freq 100 --summary_freq 10 --output_dir /farmshare/user_data/${USER}/TEGAN/${TASK}/output${SUFFIX} --summary_dir /farmshare/user_data/${USER}/TEGAN/${TASK}/summary${SUFFIX} --log_file /farmshare/user_data/${USER}/TEGAN/${TASK}/log${SUFFIX}.dat"

echo ${CMD}

${CMD}
