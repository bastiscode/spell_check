#!/bin/bash

script_dir=$(dirname "$0")
config_dir=$(realpath "$script_dir"/../configs)
workspace=$(realpath "$script_dir"/../..)

cd "$workspace" || exit 1

# some defaults which stay the same across approaches
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
export NSC_MIXED_PRECISION=true

approach=${APPROACH?"APPROACH is not defined"}

if [[ $approach == "gnn" ]]; then
  declare -a approach_env_vars=(
    "NSC_ADD_WORD_FEATURES"
    )
  declare -a approach_num_nodes=(
    "2"
    "3"
  )
  declare -a approaches=(
    "false"
    "true"
  )
  declare -a approach_names=(
    "gnn_no_feat"
    "gnn_cliques_wfc"
  )

  for approach_idx in ${!approaches[@]}; do
    declare -a approach=(${approaches[$approach_idx]})
    export NSC_EXPERIMENT_NAME=${approach_names[$approach_idx]}
    export NSC_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")

    config="$config_dir/train/sed_sequence.yaml"
    rel_config=$(realpath "$config" --relative-to "$workspace")
    export NSC_CONFIG=$rel_config
    echo "Starting approach with config $(realpath $config --relative-to $config_dir) and parameters:"

    for idx in ${!approach[@]}; do
      env_var=${approach_env_vars[$idx]}
      val=${approach[$idx]}
      echo $env_var=$val
      if [[ $val != "null" ]]; then
        export $env_var=$val
      fi
    done

    echo ""

    num_nodes=${approach_num_nodes[$approach_idx]}
    export NSC_WORLD_SIZE=$((4 * num_nodes))
    sbatch --nodes=$num_nodes spell_checking/scripts/train.sh

  done

elif [[ $approach == "transformer" ]]; then
  declare -a approach_env_vars=(
    "NSC_ADD_WORD_FEATURES"
  )
  declare -a approaches=(
    "true"
    "false"
  )
  declare -a approach_names=(
    "transformer"
    "transformer_no_feat"
  )

  for approach_idx in ${!approaches[@]}; do
    declare -a approach=(${approaches[$approach_idx]})
    export NSC_EXPERIMENT_NAME=${approach_names[$approach_idx]}
    export NSC_MASTER_PORT=$(python -c "import random; print(random.randrange(10000, 60000))")

    config="$config_dir/train/sed_sequence_transformer.yaml"
    rel_config=$(realpath "$config" --relative-to "$workspace")
    export NSC_CONFIG=$rel_config
    echo "Starting approach with config $(realpath $config --relative-to $config_dir) and parameters:"

    for idx in ${!approach[@]}; do
      env_var=${approach_env_vars[$idx]}
      val=${approach[$idx]}
      echo $env_var=$val
      if [[ $val != "null" ]]; then
        export $env_var=$val
      fi
    done

    echo ""

    export NSC_WORLD_SIZE=8
    sbatch --nodes=2 spell_checking/scripts/train.sh

  done

else
  echo "Unknown approach type $approach"

fi
