#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across ablations
export GNN_LIB_EPOCHS=1
export GNN_LIB_HIDDEN_DIM=512
export GNN_LIB_NUM_LAYERS=6
export GNN_LIB_NUM_DECODER_LAYERS=6
export GNN_LIB_LOG_PER_EPOCH=200
export GNN_LIB_EVAL_PER_EPOCH=20
export GNN_LIB_DATA_LIMIT=50000000
export GNN_LIB_LR=0.0001
export GNN_LIB_WEIGHT_DECAY=0.01
export GNN_LIB_MIXED_PRECISION=true
export GNN_LIB_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")

ablations_type=${ABLATIONS_TYPE:-"ABLATIONS_TYPE is not defined"}

if [[ $ablations_type == "transformer" ]]; then
  config="$config_dir/sec_words_nmt_transformer.yaml"
  rel_config=$(realpath "$config" --relative-to "$workspace")
  echo "Starting ablation with config $(realpath "$config" --relative-to "$config_dir")"
  export GNN_LIB_EXPERIMENT_NAME="transformer_sec_words_nmt"
  export GNN_LIB_BATCH_MAX_LENGTH=32768
  export GNN_LIB_CONFIG=$rel_config
  sbatch spelling_correction/scripts/train.sh
else
  echo "Unknown ablation type $ablations_type"
fi
