#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across ablations
export GNN_LIB_EPOCHS=1
export GNN_LIB_HIDDEN_DIM=512
export GNN_LIB_NUM_LAYERS=12
export GNN_LIB_LOG_PER_EPOCH=200
export GNN_LIB_EVAL_PER_EPOCH=20
export GNN_LIB_DATA_LIMIT=100000000
export GNN_LIB_LR=0.0001
export GNN_LIB_WEIGHT_DECAY=0.01
export GNN_LIB_MIXED_PRECISION=true
export GNN_LIB_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")

approach=${APPROACH:-"APPROACH is not defined"}

# architecture ablations
if [[ $approach == "encoder_only" ]]; then
  config="$config_dir/tokenization_repair.yaml"
  rel_config=$(realpath "$config" --relative-to "$workspace")
  echo "Starting training with config $(realpath "$config" --relative-to "$config_dir")"
  export GNN_LIB_EXPERIMENT_NAME="transformer_encoder_only"
  export GNN_LIB_BATCH_MAX_LENGTH=32768
  export GNN_LIB_CONFIG=$rel_config
  sbatch spelling_correction/scripts/train.sh
else
  echo "Unknown approach $approach"
fi
