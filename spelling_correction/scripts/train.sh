#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=training
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

master_port=${GNN_LIB_MASTER_PORT:-33334}

force_local=${GNN_LIB_FORCE_LOCAL:-false}
if [[ -n $SLURM_JOB_ID && $force_local == false ]] ; then
  script_dir=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
  is_local=false
else
  script_dir=$(realpath $0)
  is_local=true
fi
script_dir=$(dirname $script_dir)
echo "Script is located at $script_dir"

if [[ $is_local == true ]]; then
  workspace=$(realpath $script_dir/../..)
  cd $workspace
  data_dir=$workspace/data
  experiment_dir=$workspace/local_experiments

  master_addr="127.0.0.1"
  world_size=$(python -c "import torch; print(torch.cuda.device_count())")
else
  export MPLCONFIGDIR=$TMPDIR/matplotlib
  export GNN_LIB_DISABLE_TQDM=true
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_IB_DISABLE=1

  workspace=/work/dlclarge1/swalter-masters_thesis/masters_thesis
  cd $workspace
  source ../env/bin/activate

#  sync_datasets=${SYNC_DATASETS?"Could not find SYNC_DATASETS env variable"}
#  srun $script_dir/sync_data.sh $workspace $TMPDIR "$sync_datasets"
#  data_dir=$TMPDIR/data

  data_dir=$workspace/data
  experiment_dir=$workspace/experiments

  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  world_size=12
  echo "Running on Slurm Cluster, master machine at $master_addr:$master_port"
fi

# for gnn_lib
export GNN_LIB_CONFIG_DIR=$workspace/spelling_correction/configs
export GNN_LIB_DATA_DIR=$data_dir
export GNN_LIB_EXPERIMENT_DIR=$experiment_dir

echo "config dir: $GNN_LIB_CONFIG_DIR, data_dir: $GNN_LIB_DATA_DIR, experiment_dir: $GNN_LIB_EXPERIMENT_DIR"

# for pytorch distributed
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$world_size

config=${GNN_LIB_CONFIG:-""}
resume=${GNN_LIB_RESUME:-""}

if [[ ($config == "" && $resume == "") || ($config != "" && $resume != "") ]]; then
  echo "Specify either GNN_LIB_CONFIG or GNN_LIB_RESUME, but not both or neither"
  exit 1
fi

if [[ $config != "" ]]; then
  train_cmd="gnn_lib/train.py --config $config"
else
  train_cmd="gnn_lib/train.py --resume $resume"
fi

echo "GPU information:"
nvidia-smi

if [[ $is_local == true ]]; then
  echo "Starting local training with cmd $train_cmd"
  torchrun \
    --nnodes=1 \
    --nproc_per_node=$world_size \
    $train_cmd
else
  echo "Starting Slurm training with cmd $train_cmd"
  srun python $train_cmd
fi
