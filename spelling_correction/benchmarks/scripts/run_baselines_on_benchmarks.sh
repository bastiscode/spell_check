#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=run_baselines
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

script_dir=$(dirname $0)

split=${SPLIT:-test}
benchmark_dir=$(realpath $script_dir/../$split)
baselines_dir=$(realpath $script_dir/../../baselines)

benchmarks="sed_sequence sed_words sec"
sec_baselines="SEC_ASPELL SEC_JAMSPELL SEC_DUMMY SEC_CTD SEC_LANGUAGETOOL SEC_NEUSPELL_BERT"
sed_from_sec_baselines="SEC_ASPELL SEC_JAMSPELL SEC_DUMMY SEC_CTD SEC_LANGUAGETOOL SEC_NEUSPELL_BERT"
declare -A baselines=(
  ["sec"]=$sec_baselines
  ["sed_sequence"]="SED_SEQUENCE_FROM_SEC SED_SEQUENCE_OOD"
  ["sed_words"]="SED_WORDS_FROM_SEC SED_WORDS_OOD"
)

baseline_regex=${BASELINE_REGEX:-"*"}
benchmark_regex=${BENCHMARK_REGEX:-"*"}

for benchmark in ${benchmarks[@]}
do
  for in_file in `ls $benchmark_dir/$benchmark/*/*/corrupt.txt`
  do
    out_dir=$(dirname `realpath --relative-to=$benchmark_dir/$benchmark $in_file`)

    # skip 1blm benchmarks for now since they take too long
    if [[ $out_dir == *"1blm_"* || $in_file != $benchmark_regex ]]
    then
      continue
    fi

    for baseline in ${baselines[$benchmark]}
    do
      if [[ $baseline == *_FROM_SEC ]]
      then
        for sec_baseline in ${sed_from_sec_baselines[@]}
        do
          if [[ $sec_baseline != $baseline_regex ]]; then
            echo "sec baseline $sec_baseline not matching regex, skipping"
            continue
          fi
          echo "Running baseline $baseline ($sec_baseline) on $out_dir of benchmark $benchmark"
          python $baselines_dir/run.py \
            --in-file $in_file \
            --out-dir $benchmark_dir/$benchmark/results/$out_dir \
            --baseline $baseline \
            --sec-baseline=$sec_baseline \
            --batch-size ${BATCH_SIZE:-16} \
            --sort-by-length
        done
      else
        if [[ $baseline != $baseline_regex ]]; then
            echo "baseline $baseline not matching regex, skipping"
            continue
        fi
        echo "Running baseline $baseline on $out_dir of benchmark $benchmark"
        python $baselines_dir/run.py \
          --in-file $in_file \
          --out-dir $benchmark_dir/$benchmark/results/$out_dir \
          --baseline $baseline \
          --sec-baseline=$sec_baseline \
          --batch-size ${BATCH_SIZE:-16} \
          --sort-by-length
      fi
    done
  done
done
