#!/bin/bash

script_dir=$(realpath $(dirname $0))

if [[ -x `command -v rustc` ]]; then
  echo "Found Rust toolchain"
else
  echo "Installing Rust toolchain"
  curl -sSf https://sh.rustup.rs | bash -s -- -y
  source $HOME/.cargo/env
fi
echo "Rust version: $(rustc --version)"

echo "Installing maturin"
make -C $script_dir/../utils/edit_distance_rs setup

echo "Compiling and installing Rust extensions"
make -C $script_dir/../utils/edit_distance_rs install
