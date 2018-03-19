#!/bin/bash
#SBATCH --job-name=TEGAN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --output=TEGAN-%j.out
#SBATCH --error=TEGAN-%j.err
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

cd /home/${USER}/Codes/TEGAN

# Setup the environment
module purge
module load cuda/9.0 cudnn/7.0.3
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
source ~/tensorFlowGpu/bin/activate

TASK="TEGAN"
LEARNING_RATE=0.0001
DECAY_STEP=600
DECAY_RATE=0.5
STAIR="True"
RESBLOCKS=12
GENFREQ=2
GENSTART=300
ADVERSARIAL_RATIO=0.01
ENS=0.2
CON=0.5
PHY=0.125

SUFFIX_RESNET="_RB${RESBLOCKS}_LR${LEARNING_RATE}_ENS${ENS}_CON${CON}_PHY${PHY}"
SUFFIX="_RB${RESBLOCKS}_LR${LEARNING_RATE}_ENS${ENS}_CON${CON}_PHY${PHY}_GS${GENSTART}_GF${GENFREQ}_AR${ADVERSARIAL_RATIO}_DR${DECAY_RATE}_DS${DECAY_STEP}_STR-${STAIR}"

CMD="python main.py \
--task ${TASK} \
--num_resblock ${RESBLOCKS} \
--learning_rate ${LEARNING_RATE} \
--decay_rate ${DECAY_RATE} \
--decay_step ${DECAY_STEP} \
--stair ${STAIR} \
--gen_start ${GENSTART} \
--gen_freq ${GENFREQ} \
--adversarial_ratio ${ADVERSARIAL_RATIO} \
--lambda_ens ${ENS} \
--lambda_con ${CON} \
--lambda_phy ${PHY} \
--max_iter 10000 \
--max_epoch 50 \
--save_freq 100 \
--summary_freq 10 \
--train_dir /farmshare/user_data/${USER}/TEGAN/Data/train \
--dev_dir /farmshare/user_data/${USER}/TEGAN/Data/dev \
--output_dir /farmshare/user_data/${USER}/TEGAN/${TASK}/output${SUFFIX} \
--summary_dir /farmshare/user_data/${USER}/TEGAN/${TASK}/summary${SUFFIX} \
--log_file /farmshare/user_data/${USER}/TEGAN/${TASK}/log${SUFFIX}.dat \
--checkpoint /farmshare/user_data/${USER}/TEGAN/TEResNet/output${SUFFIX_RESNET}/model-9200 \
--pre_trained_generator True "


echo ${CMD}

${CMD} >& log${SUFFIX}.out
