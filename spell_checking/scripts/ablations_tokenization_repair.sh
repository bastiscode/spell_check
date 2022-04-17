#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across ablations
export NSC_EPOCHS=1
export NSC_HIDDEN_DIM=512
export NSC_NUM_INPUT_LAYERS=6
export NSC_NUM_WORD_LAYERS=6
export NSC_LOG_PER_EPOCH=200
export NSC_EVAL_PER_EPOCH=20
export NSC_DATA_LIMIT=100000000
export NSC_LR=0.0001
export NSC_WEIGHT_DECAY=0.01
export NSC_MIXED_PRECISION=true

approach=${APPROACH:-"APPROACH is not defined"}

# architecture ablations
if [[ $approach == "encoder_only" ]]; then
  config="$config_dir/tokenization_repair.yaml"
  rel_config=$(realpath "$config" --relative-to "$workspace")
  echo "Starting training with config $(realpath "$config" --relative-to "$config_dir")"
  export NSC_EXPERIMENT_NAME="transformer_encoder_only"
  export NSC_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")
  export NSC_BATCH_MAX_LENGTH=98304
  export NSC_CONFIG=$rel_config
  sbatch spell_checking/scripts/train.sh
else
  echo "Unknown approach $approach"
fi
