#!/bin/bash

workspace=$1
tmpdir=$2
datasets=$3

if [[ $SLURM_LOCALID != 0 ]]; then
  exit 0
else
  echo "[process $SLURM_PROCID, node $SLURM_NODEID] syncing data"
  echo "[process $SLURM_PROCID, node $SLURM_NODEID] workspace: $workspace, tmpdir: $tmpdir, datasets: $datasets"
  mkdir -p $tmpdir/data/processed
  for dataset in ${datasets[@]}; do
    echo "[process $SLURM_PROCID, node $SLURM_NODEID] syncing dataset $dataset"
    rsync -ah $workspace/data/processed/$dataset $tmpdir/data/processed
  done
  echo "[process $SLURM_PROCID, node $SLURM_NODEID] syncing tokenizers, dictionaries, etc."
  rsync -ah $workspace/data/spell_check_index $tmpdir/data
  rsync -ah $workspace/data/tokenizers $tmpdir/data
  rsync -ah $workspace/data/dictionaries $tmpdir/data
  rsync -ah $workspace/data/misspellings $tmpdir/data
  ls -lah $tmpdir/data/*
fi
