#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --mincpus=32
#SBATCH --mem=256G
#SBATCH --job-name=preprocessing
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

config=${GNN_LIB_CONFIG?"GNN_LIB_CONFIG env variable not found"}

export GNN_LIB_CONFIG_DIR=$workspace/spelling_correction/configs
export GNN_LIB_DATA_DIR=$data_dir

python gnn_lib/preprocess.py $config
