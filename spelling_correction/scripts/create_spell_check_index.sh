#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --mincpus=32
#SBATCH --mem=256G
#SBATCH --job-name=create_spell_check_index
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

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

python gnn_lib/scripts/create_spell_check_index.py \
	--in-file data/cleaned/wikidump_paragraphs/train_files.txt data/cleaned/bookcorpus_paragraphs/train_files.txt \
	--context-length 1 \
	--out-dir data/spell_check_index/ctx_1_ned_string \
	--dist norm_edit_distance \
	--vectorizer string \
	--dictionary-file data/dictionaries/merged_train_100k.txt \
	--max-files 2000 \
	--min-freq 2

python gnn_lib/scripts/create_spell_check_index.py \
	--in-file data/cleaned/wikidump_paragraphs/train_files.txt data/cleaned/bookcorpus_paragraphs/train_files.txt \
	--context-length 1 \
	--out-dir data/spell_check_index/ctx_1_euclidean_custom \
	--dist euclidean \
	--vectorizer custom \
	--vectorizer-path data/spell_check_index/custom_vectorizers/context_char_transformer.pt \
	--dictionary-file data/dictionaries/merged_train_100k.txt \
	--max-files 2000 \
	--min-freq 2

python gnn_lib/scripts/create_spell_check_index.py \
	--in-file data/cleaned/wikidump_paragraphs/train_files.txt data/cleaned/bookcorpus_paragraphs/train_files.txt \
	--context-length 2 \
	--out-dir data/spell_check_index/ctx_2_ned_string \
	--dist norm_edit_distance \
	--vectorizer string \
	--dictionary-file data/dictionaries/merged_train_100k.txt \
	--max-files 2000 \
	--min-freq 2

python gnn_lib/scripts/create_spell_check_index.py \
	--in-file data/cleaned/wikidump_paragraphs/train_files.txt data/cleaned/bookcorpus_paragraphs/train_files.txt \
	--context-length 2 \
	--out-dir data/spell_check_index/ctx_2_euclidean_custom \
	--dist euclidean \
	--vectorizer custom \
	--vectorizer-path data/spell_check_index/custom_vectorizers/context_char_transformer.pt \
	--dictionary-file data/dictionaries/merged_train_100k.txt \
	--max-files 2000 \
	--min-freq 2
