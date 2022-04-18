#!/bin/bash

script_dir=$(dirname $0)
base_dir=$(realpath $script_dir/../../..)
data_dir=$base_dir/data/cleaned
misspellings_dir=$base_dir/data/misspellings
benchmark_dir=$(realpath $script_dir/..)
datasets=("bookcorpus_paragraphs" "wikidump_paragraphs")
dataset_names=("bookcorpus" "wikidump")
misspelling_types="artificial realistic"
output_types="sed_sequence sed_words sec"
max_sequences=10000

for idx in ${!datasets[@]}
do
  dataset=${datasets[$idx]}
  dataset_name=${dataset_names[$idx]}
  for misspelling_type in ${misspelling_types[@]}
  do
    for output_type in ${output_types[@]}
    do
      # test benchmarks with test misspellings
      python $script_dir/create_spelling_benchmarks.py \
      --in-file $data_dir/$dataset/test_files.txt \
      --out-dir $benchmark_dir/test \
      --misspellings-dir $misspellings_dir \
      --misspelling-type $misspelling_type \
      --misspelling-split test \
      --output-type $output_type \
      --max-sequences $max_sequences \
      --benchmark-name $dataset_name

      # dev benchmarks with dev misspellings
      python $script_dir/create_spelling_benchmarks.py \
      --in-file $data_dir/$dataset/dev_files.txt \
      --out-dir $benchmark_dir/dev \
      --misspellings-dir $misspellings_dir \
      --misspelling-type $misspelling_type \
      --misspelling-split dev \
      --output-type $output_type \
      --max-sequences $max_sequences \
      --benchmark-name $dataset_name
    done
  done
done
