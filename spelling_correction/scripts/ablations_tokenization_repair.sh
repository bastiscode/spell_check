#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across ablations
export GNN_LIB_EPOCHS=4
export GNN_LIB_HIDDEN_DIM=512
export GNN_LIB_LOG_PER_EPOCH=200
export GNN_LIB_EVAL_PER_EPOCH=10
# train limit is per dataset
export GNN_LIB_TRAIN_LIMIT=20000000
# val limit is per dataset
export GNN_LIB_VAL_LIMIT=5000
export GNN_LIB_NUM_WORKERS=8
export GNN_LIB_LR=0.0001
export GNN_LIB_WEIGHT_DECAY=0.01
export GNN_LIB_PREPROCESS_NAME="tokenization_repair_with_ned_index"
export GNN_LIB_NUM_NEIGHBORS=3
export GNN_LIB_BATCH_MAX_LENGTH=4096
export GNN_LIB_USE_MIXED_PREC="true"

# list space delimited datasets to sync to temp node directory here
#export SYNC_DATASETS="NEUSPELL WIKIDUMP BOOKCORPUS"

approach=${APPROACH:-"APPROACH is not defined"}

# architecture ablations
if [[ $approach == "encoder_only" ]]; then
  config="$config_dir/tokenization_repair.yaml"
  rel_config=$(realpath "$config" --relative-to "$workspace")
  echo "Starting training with config $(realpath "$config" --relative-to "$config_dir")"
  export GNN_LIB_EXPERIMENT_NAME="transformer_encoder_only"
  export GNN_LIB_NUM_LAYERS=6
  export GNN_LIB_CONFIG=$rel_config
  sbatch spelling_correction/scripts/train.sh
elif [[ $approach == "nmt" ]]; then
  config="$config_dir/tokenization_repair_nmt.yaml"
  rel_config=$(realpath "$config" --relative-to "$workspace")
  echo "Starting training with config $(realpath "$config" --relative-to "$config_dir")"
  export GNN_LIB_EXPERIMENT_NAME="transformer_nmt"
  export GNN_LIB_NUM_LAYERS=3
  export GNN_LIB_NUM_DECODER_LAYERS=3
  export GNN_LIB_CONFIG=$rel_config
  sbatch spelling_correction/scripts/train.sh
  exit
else
  echo "Unknown approach $approach"
fi
