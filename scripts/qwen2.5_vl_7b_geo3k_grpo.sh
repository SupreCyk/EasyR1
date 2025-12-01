#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3

# MODEL_PATH=/home/chenyukun/workspace/data2/chenyukun/models/HF-Models/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
# export RAY_TMPDIR=/data2/chenyukun/workspace/ray_tmp
# export TENSORBOARD_DIR="/data2/chenyukun/workspace/tensorboard/"

MODEL_PATH=/home/runjin.cyk/models/Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path
export TENSORBOARD_DIR="/home/runjin.cyk/ossbucket/oss_bucket_0/tensorboard/easyR1/${EXPERIMENT_NAME}"


mkdir -p ${TENSORBOARD_DIR}

PROJECT_NAME=easy_r1  # 从 config.yaml 中获取的默认值
EXPERIMENT_NAME=qwen2_5_vl_7b_geo3k_grpo_disable_tqdm


# 构建实际的实验路径
EXPERIMENT_DIR="${TENSORBOARD_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p ${EXPERIMENT_DIR}
cp $0 ${EXPERIMENT_DIR}/run.sh

# python3 -m verl.trainer.main \
#     config=examples/config.yaml \
#     data.train_files=/data2/chenyukun/workspace/V/datas/geo3k_hg/train-indexed-00000-of-00001.parquet \
#     data.val_files=/data2/chenyukun/workspace/V/datas/geo3k_hg/test-indexed-00000-of-00001.parquet \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.project_name=${PROJECT_NAME}\
#     trainer.experiment_name=${EXPERIMENT_NAME}\
#     trainer.n_gpus_per_node=4\
#     worker.actor.fsdp.torch_dtype=bf16 \
#     worker.ref.fsdp.torch_dtype=bf16 \
#     trainer.save_checkpoint_path=/data2/chenyukun/workspace/V/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}\
#     trainer.logger='[file, tensorboard]'\
#     trainer.rollout_data_dir=/data2/chenyukun/workspace/V/rollout_data/${PROJECT_NAME}/${EXPERIMENT_NAME} \
#     $@ 2>&1 | tee "verl_qwen2_5_vl_7b_geo3k_grpo_batchsize128_n8.log"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/runjin.cyk/workspace/hiyouga/geometry3k/data/train-indexed-00000-of-00001.parquet \
    data.val_files=/home/runjin.cyk/workspace/hiyouga/geometry3k/data/test-indexed-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name=${PROJECT_NAME}\
    trainer.experiment_name=${EXPERIMENT_NAME}\
    worker.rollout.disable_tqdm=true\
    trainer.n_gpus_per_node=4\
    trainer.save_checkpoint_path=/home/runjin.cyk/ossbucket/oss_bucket_0/checkpoints/easyr1/${EXPERIMENT_NAME}\
    trainer.rollout_data_dir=/home/runjin.cyk/ossbucket/oss_bucket_0/rolloutdatas/easyr1/${EXPERIMENT_NAME}\
    trainer.logger='[file, tensorboard]'\
    $@ 2>&1 | tee "verl_qwen2_5_vl_7b_geo3k_grpo_batchsize128_n8_disable_tqdm.log"

