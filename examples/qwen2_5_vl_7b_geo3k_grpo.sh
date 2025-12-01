#!/bin/bash

set -x

MODEL_PATH=/home/chenyukun/workspace/data2/chenyukun/models/HF-Models/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
export RAY_TMPDIR=/data2/chenyukun/workspace/ray_tmp
export TENSORBOARD_DIR="/data2/chenyukun/workspace/tensorboard/v2/easyR1/qwen2_5_vl_7b_geo3k_grpo/tensorboard_logs"

mkdir -p ${TENSORBOARD_DIR}
cp $0 ${TENSORBOARD_DIR}/run.sh

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/chenyukun/datasets/geometry3k/data/train-00000-of-00001.parquet \
    data.val_files=/home/chenyukun/datasets/geometry3k/data/test-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo_new_dataset \
    trainer.n_gpus_per_node=8\
    trainer.logger='[file, tensorboard]'\
    $@ 2>&1 | tee "verl_qwen2.5_vl_7b_grpo_rule_base_new_dataset.log"