#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across ablations
export GNN_LIB_EPOCHS=1
export GNN_LIB_HIDDEN_DIM=512
export GNN_LIB_EDGE_HIDDEN_DIM=256
export GNN_LIB_NUM_LAYERS=6
export GNN_LIB_LOG_PER_EPOCH=200
export GNN_LIB_EVAL_PER_EPOCH=20
export GNN_LIB_DATA_LIMIT=50000000
export GNN_LIB_LR=0.0001
export GNN_LIB_WEIGHT_DECAY=0.01
export GNN_LIB_NUM_NEIGHBORS=3

export GNN_LIB_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")
ablations_type=${ABLATIONS_TYPE:-"ABLATIONS_TYPE is not defined"}

# architecture ablations
if [[ $ablations_type == "gnn" ]]; then
  declare -a ablation_env_vars=(
    "GNN_LIB_MESSAGE_GATING"
    "GNN_LIB_MESSAGE_SCHEME"
    "GNN_LIB_NODE_UPDATE"
    "GNN_LIB_SHARE_PARAMETERS"
    "GNN_LIB_ADD_WORD_FEATURES"
    "GNN_LIB_ADD_DEPENDENCY_INFO"
    "GNN_LIB_WORD_FULLY_CONNECTED"
    "GNN_LIB_TOKEN_FULLY_CONNECTED"
    "GNN_LIB_SPELL_CHECK_INDEX"
    )
  declare -a ablations=(
    "false attention residual false true false false true null" # default
#    "true message_passing residual false true true false true null" # default + message_gating + dep
#    "true message_passing residual false true false true false data/spell_check_index/ctx_0_ned_string" # default + message_gating + cliques + neighbors
  )
  declare -a ablation_names=(
    "gnn_default"
#    "gnn_dependency"
#    "gnn_neighbors"
  )

  for ablation_idx in ${!ablations[@]}; do
    declare -a ablation=(${ablations[$ablation_idx]})
    export GNN_LIB_EXPERIMENT_NAME=${ablation_names[$ablation_idx]}
    export GNN_LIB_BATCH_MAX_LENGTH=32768

    config="$config_dir/sed_sequence.yaml"
    rel_config=$(realpath "$config" --relative-to "$workspace")
    export GNN_LIB_CONFIG=$rel_config
    echo "Starting ablation with config $(realpath $config --relative-to $config_dir) and parameters:"

    for idx in ${!ablation[@]}; do
      env_var=${ablation_env_vars[$idx]}
      val=${ablation[$idx]}
      # turn off mixed precision with convolutional architecture because its not supported
      if [[ $env_var == "GNN_LIB_MESSAGE_SCHEME" && $val == "convolution" ]]; then
        export GNN_LIB_MIXED_PRECISION=false
      else
        export GNN_LIB_MIXED_PRECISION=true
      fi
      echo $env_var=$val
      if [[ $val != "null" ]]; then
        export $env_var=$val
      fi
    done

    echo ""

    sbatch spelling_correction/scripts/train.sh
  done

elif [[ $ablations_type == "transformer" ]]; then
  config="$config_dir/sed_sequence_transformer.yaml"
  rel_config=$(realpath "$config" --relative-to "$workspace")
  echo "Starting ablation with config $(realpath "$config" --relative-to "$config_dir")"
  export GNN_LIB_EXPERIMENT_NAME="transformer"
  export GNN_LIB_BATCH_MAX_LENGTH=32768
  export GNN_LIB_MIXED_PRECISION=true
  export GNN_LIB_CONFIG=$rel_config
  sbatch spelling_correction/scripts/train.sh

else
  echo "Unknown ablation type $ablations_type"

fi
