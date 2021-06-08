#!/bin/bash

cd /home/manlong/IPython_Notebook/TEGAN

# Setup the environment
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

TASK="TEResNet"
LEARNING_RATE=0.0001
RESBLOCKS=12
ENS=0.2
CON=0.5
PHY=0.250
# PHY=0.125
SUFFIX="_RB${RESBLOCKS}_LR${LEARNING_RATE}_ENS${ENS}_CON${CON}_PHY${PHY}"

CMD="python3 main.py \
--task ${TASK} \
--num_resblock ${RESBLOCKS} \
--learning_rate ${LEARNING_RATE} \
--lambda_ens ${ENS} \
--lambda_con ${CON} \
--lambda_phy ${PHY} \
--max_iter 10000 \
--max_epoch 50 \
--save_freq 100 \
--summary_freq 10 \
--train_dir    /home/manlong/Data/HIT/output/64x64x64/TFRecords/train \
--dev_dir      /home/manlong/Data/HIT/output/64x64x64/TFRecords/test \
--output_dir   /home/manlong/IPython_Notebook/TEGAN/short_lambda_p_0_250/${TASK}/output${SUFFIX} \
--summary_dir  /home/manlong/IPython_Notebook/TEGAN/short_lambda_p_0_250/${TASK}/summary${SUFFIX} \
--log_file     /home/manlong/IPython_Notebook/TEGAN/short_lambda_p_0_250/${TASK}/log${SUFFIX}.dat \
--pre_trained_model False "


echo ${CMD}

${CMD}
