#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=run_experiments
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

script_dir=$(dirname $0)

split=${SPLIT:-test}
benchmark_dir=$(realpath $script_dir/../$split)
base_dir=$(realpath $script_dir/../../..)
cd "$base_dir" || exit 1
gnn_lib_dir=$(realpath $base_dir/gnn_lib)
bin_dir=$(realpath $base_dir/bin)
experiments_dir=$(realpath $base_dir/experiments)
data_dir=$(realpath $base_dir/data)
config_dir=$(realpath $base_dir/spelling_correction/configs)

# some configuration options
overwrite=${OVERWRITE:-false}

declare -A experiment_to_benchmark=(
  ["TOKENIZATION_REPAIR"]="tokenization_repair"
  ["SED_WORDS"]="sed_words"
  ["SED_SEQUENCE"]="sed_sequence"
  ["SEC_NMT"]="sec"
  ["SEC_WORDS_NMT"]="sec"
)

declare -A experiment_to_exec=(
  ["TOKENIZATION_REPAIR"]="trt"
  ["SED_WORDS"]="gsed"
  ["SED_SEQUENCE"]="gsed"
  ["SEC_NMT"]="gsec"
  ["SEC_WORDS_NMT"]="gsec"
)

exp_regex=${EXP_REGEX:-"*"}
benchmark_regex=${BENCHMARK_REGEX:-"*"}

for experiment in $experiments_dir/*/*
do
  experiment_rel=$(realpath --relative-to=$experiments_dir $experiment)
  experiment_type=$(echo $experiment_rel | cut -d "/" -f1)
  experiment_name=${experiment_rel//\//_}

  if [[ $experiment != $exp_regex ]]; then
    echo "experiment $experiment_name not matching regex $exp_regex, skipping"
    continue
  fi

  if [[ ! -f $experiment/checkpoints/checkpoint_best.pt ]]; then
    echo "experiment $experiment_rel has no best checkpoint file"
    continue
  fi

  echo "Enter model name for experiment $experiment_name (or type 'skip' to skip):"

  read model_name
  if [[ "$model_name" == "skip" ]]; then
    echo "Skipping experiment $experiment_name"
    continue
  fi

  bin_name=${experiment_to_exec[$experiment_type]}

  for benchmark in ${experiment_to_benchmark[$experiment_type]}
  do
    in_files=`ls $benchmark_dir/$benchmark/*/*/corrupt.txt`
    for in_file in $in_files
    do
      if [[ $in_file != $benchmark_regex ]]; then
        echo "benchmark $in_file not matching regex $benchmark_regex, skipping"
        continue
      fi

      out_dir_rel=$(dirname $(realpath --relative-to=$benchmark_dir/$benchmark $in_file))
      out_dir=$benchmark_dir/$benchmark/results/$out_dir_rel

      # skip 1blm benchmarks and bea60k for now since they take too long
      if [[ $out_dir_rel == *"1blm_"* ]]
      then
        continue
      fi

      echo "Running experiment $experiment_name ($experiment_type) on $out_dir_rel of $benchmark benchmark"
      ${bin_dir}/${bin_name} -e $experiment -f $in_file -o $out_dir/${model_name}.txt -b 128
    done
  done
done
