#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across ablations
export NSC_EPOCHS=1
export NSC_HIDDEN_DIM=512
export NSC_EDGE_HIDDEN_DIM=256
export NSC_NUM_LAYERS=6
export NSC_LOG_PER_EPOCH=200
export NSC_EVAL_PER_EPOCH=20
export NSC_BATCH_MAX_LENGTH=98304
export NSC_DATA_LIMIT=100000000
export NSC_LR=0.0001
export NSC_WEIGHT_DECAY=0.01
export NSC_NUM_NEIGHBORS=3
export NSC_MIXED_PRECISION=true

ablations_type=${ABLATIONS_TYPE:-"ABLATIONS_TYPE is not defined"}

# architecture ablations
if [[ $ablations_type == "gnn" ]]; then
  declare -a ablation_env_vars=(
    "NSC_MESSAGE_GATING"
    "NSC_MESSAGE_SCHEME"
    "NSC_NODE_UPDATE"
    "NSC_SHARE_PARAMETERS"
    "NSC_ADD_WORD_FEATURES"
    "NSC_ADD_DEPENDENCY_INFO"
    "NSC_WORD_FULLY_CONNECTED"
    "NSC_TOKEN_FULLY_CONNECTED"
    "NSC_SPELL_CHECK_INDEX"
    )
  declare -a ablations=(
#    "false attention residual false false false true false null" # no_feat
    "false attention residual false true false true false null" # default
  )
  declare -a ablation_names=(
#    "gnn_no_feat"
    "gnn_cliques_wfc"
  )

  for ablation_idx in ${!ablations[@]}; do
    declare -a ablation=(${ablations[$ablation_idx]})
    export NSC_EXPERIMENT_NAME=${ablation_names[$ablation_idx]}
    export NSC_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")

    config="$config_dir/sed_sequence.yaml"
    rel_config=$(realpath "$config" --relative-to "$workspace")
    export NSC_CONFIG=$rel_config
    echo "Starting ablation with config $(realpath $config --relative-to $config_dir) and parameters:"

    for idx in ${!ablation[@]}; do
      env_var=${ablation_env_vars[$idx]}
      val=${ablation[$idx]}
      echo $env_var=$val
      if [[ $val != "null" ]]; then
        export $env_var=$val
      fi
    done

    echo ""

    sbatch spell_checking/scripts/train.sh
  done

elif [[ $ablations_type == "transformer" ]]; then
  declare -a ablation_env_vars=(
    "NSC_ADD_WORD_FEATURES"
    "NSC_DICTIONARY"
  )
  declare -a ablations=(
    "true null"
    "false null"
  )
  declare -a ablation_names=(
    "transformer"
    "transformer_no_feat"
  )

  for ablation_idx in ${!ablations[@]}; do
    declare -a ablation=(${ablations[$ablation_idx]})
    export NSC_EXPERIMENT_NAME=${ablation_names[$ablation_idx]}
    export NSC_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")

    config="$config_dir/sed_sequence_transformer.yaml"
    rel_config=$(realpath "$config" --relative-to "$workspace")
    export NSC_CONFIG=$rel_config
    echo "Starting ablation with config $(realpath $config --relative-to $config_dir) and parameters:"

    for idx in ${!ablation[@]}; do
      env_var=${ablation_env_vars[$idx]}
      val=${ablation[$idx]}
      echo $env_var=$val
      if [[ $val != "null" ]]; then
        export $env_var=$val
      fi
    done

    echo ""

    sbatch spell_checking/scripts/train.sh
  done

else
  echo "Unknown ablation type $ablations_type"

fi
