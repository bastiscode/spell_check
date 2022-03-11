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
experiments_dir=$(realpath $base_dir/experiments)
data_dir=$(realpath $base_dir/data)
config_dir=$(realpath $base_dir/spelling_correction/configs)

# some configuration options
overwrite=${OVERWRITE:-false}

declare -A experiment_to_benchmark=(
  ["TOKENIZATION_REPAIR"]="tokenization_repair"
  ["TOKENIZATION_REPAIR_GROUPS"]="tokenization_repair"
  ["SED_WORDS"]="sed_words"
  ["SED_SEQUENCE"]="sed_sequence"
  ["SEC_NMT"]="sec"
  ["SEC_WORDS"]="sec"
  ["SEC_WORDS_NMT"]="sec"
)

#declare -A inference_kwargs=(
#  ["threshold"]="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
#  ["inference_mode"]="greedy beam sample best_first"
#)

exp_regex=${EXP_REGEX:-"*"}
benchmark_regex=${BENCHMARK_REGEX:-"*"}

for experiment in $experiments_dir/*/*
do
  experiment_rel=$(realpath --relative-to=$experiments_dir $experiment)
  experiment_type=$(echo $experiment_rel | cut -d "/" -f1)
  experiment_name=${experiment_rel//\//_}

  if [[ $experiment != $exp_regex ]]; then
    echo "experiment not matching regex $exp_regex, skipping"
    continue
  fi

  if [[ ! -f $experiment/checkpoints/checkpoint_best.pt ]]; then
    echo "experiment $experiment_rel has no best checkpoint file"
    continue
  fi

  for benchmark in ${experiment_to_benchmark[$experiment_type]}
  do
    runtime_file=$benchmark_dir/${benchmark}_runtimes.tsv
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
      python $gnn_lib_dir/test.py \
        $experiment \
        --data-dir $data_dir \
        --config-dir $config_dir \
        --in-file $in_file \
        --out-path $out_dir \
        --batch-size ${BATCH_SIZE:-16} \
        --runtime-file $runtime_file
    done
  done
done
