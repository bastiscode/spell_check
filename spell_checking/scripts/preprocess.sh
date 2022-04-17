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
  export NSC_DISABLE_TQDM=true
  workspace=/work/dlclarge1/swalter-masters_thesis/masters_thesis
  data_dir=$workspace/data
  cd $workspace
  source ../env/bin/activate
fi

config=${NSC_CONFIG?"NSC_CONFIG env variable not found"}

export NSC_CONFIG_DIR=$workspace/spell_checking/configs
export NSC_DATA_DIR=$data_dir

python nsc/preprocess.py $config
