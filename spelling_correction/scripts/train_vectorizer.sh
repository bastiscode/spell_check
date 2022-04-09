#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_vectorizer
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00

if [[ -n $SLURM_JOB_ID ]] ; then
  script_dir=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
  is_local=false
else
  script_dir=$(realpath $0)
  is_local=true
fi
script_dir=$(dirname $script_dir)
echo "Script is located at $script_dir"

if [[ $is_local == true ]]; then
  echo "Running locally"
  workspace=$(realpath $script_dir/../..)
  data_dir=$workspace/data
else
  echo "Running on Slurm Cluster"
  export MPLCONFIGDIR=$TMPDIR/matplotlib
  export GNN_LIB_DISABLE_TQDM=true
  workspace=/work/dlclarge1/swalter-masters_thesis/masters_thesis
  data_dir=$workspace/data
  cd $workspace
  source ../env/bin/activate
fi

python gnn_lib/scripts/train_vectorizer.py \
  --train-misspelling-files data/misspellings/train_misspellings.json \
  --val-misspelling-files data/misspellings/dev_misspellings.json \
  --train-jsonl-file data/cleaned/wikidump_paragraphs/train_files.txt data/cleaned/bookcorpus_paragraphs/train_files.txt \
  --val-jsonl-file data/cleaned/wikidump_paragraphs/dev_files.txt data/cleaned/bookcorpus_paragraphs/dev_files.txt \
  --dictionary data/dictionaries/merged_train_100k.txt \
  --eval-per-epoch 10 \
  --epochs 20 \
  --out-path data/spell_check_index/custom_vectorizers/context_char_transformer.pt \
  --fp16 \
  --max-train-samples 100000000
