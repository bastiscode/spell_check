Training and reproducing results
================================

You can skip the following training section if you only want to use pretrained models and not
train your own models.

Training
--------

Before starting training you need the get the training data. Everything you need
(preprocessed samples, tokenizer, dictionaries, etc.) can be found under ``/nfs/students/sebastian-walter/masters_thesis/data``.

You also need to set the following two special environment variables:

.. code-block:: bash

    # set the data directory
    export NSC_DATA_DIR=/nfs/students/sebastian-walter/masters_thesis/data

    # set the config directory (necessary to be able to compose
    # the final training config from sub-configs)
    export NSC_CONFIG_DIR=spell_checking/configs

.. note::
    Of course you can also copy the training data folder to every other
    place you like and adjust ``NSC_DATA_DIR`` accordingly. But keep in mind that this
    folder is very large (> 1TB).

After that you can train your own models using a training config.
All of the training configs this project used to train models can be found here_.

You might have to further configure a training config by setting additional environment variables. Let's
look at an example where we want to train a spelling error detection Graph Neural Network. The `config
for this task`_ looks like the following:

.. code-block:: yaml

    variant: ${from_file:variant/sed_words.yaml}
    model: ${from_file:model/model_for_sed_words.yaml}
    optimizer: ${from_file:optimizer/adamw.yaml}
    lr_scheduler: ${from_file:lr_scheduler/step_with_0.05_warmup.yaml}

    experiment_dir: ${oc.env:NSC_EXPERIMENT_DIR}
    data_dir: ${oc.env:NSC_DATA_DIR}
    datasets:
      - ${oc.env:NSC_DATA_DIR}/processed/wikidump_paragraphs_sed_words_and_sec
      - ${oc.env:NSC_DATA_DIR}/processed/bookcorpus_paragraphs_sed_words_and_sec
    dataset_limits:
      - ${oc.env:NSC_DATA_LIMIT}
      - ${oc.env:NSC_DATA_LIMIT}
    val_splits:
      - 2500
      - 2500

    experiment_name: ${oc.env:NSC_EXPERIMENT_NAME,dummy}
    epochs: ${oc.env:NSC_EPOCHS,1}
    batch_max_length: ${oc.env:NSC_BATCH_MAX_LENGTH,4096}
    bucket_span: ${oc.env:NSC_BUCKET_SPAN,4}
    log_per_epoch: ${oc.env:NSC_LOG_PER_EPOCH,100}
    eval_per_epoch: ${oc.env:NSC_EVAL_PER_EPOCH,4}
    seed: 22
    mixed_precision: ${oc.env:NSC_MIXED_PRECISION,true}

Values looking like ``${from_file:<file_path>}`` refer to other config files relative to the ``NSC_CONFIG_DIR``. When the training
config is composed, the contents of the referred config files will replace these values.

Values looking like ``${oc.env:<env_var_name>,<default>}`` refer to environment variables and an optional default that will be set
if the environment variable is not found. If there is no default you will be required to set the environment variable, otherwise
you receive an error message.

In our example we need to set values for the environment variables ``NSC_DATA_LIMIT`` (can be used to limit the number of samples per training dataset)
and ``NSC_EXPERIMENT_DIR`` (directory path where the logs and checkpoints will be saved). Once we have set these variables we
can start the training. Since the training script is written to support distributed training we need to use torchrun_
to launch the script:

.. code-block:: bash

    # set the environment variables
    export NSC_DATA_LIMIT=100000 # set data limit to 100,000 samples per dataset
    export NSC_EXPERIMENT_DIR=experiments # directory path where the experiment will be saved

    # to train locally / on a single node
    torchrun --nnodes=1 nsc/train.py --config spell_checking/configs/sed_words.yaml

You can also resume training for an existing experiment if you had to abort training for some reason:

.. code-block:: bash

    # resume training from latest checkpoint of an experiment
    torchrun --nnodes=1 nsc/train.py --resume <path_to_experiment_directory>

As an alternative you can set one of the ``NSC_CONFIG`` or ``NSC_RESUME`` environment variables
and use the `train.sh`_ script to start training. This script additionally provides functionality to start distributed
training on SLURM_ clusters. Training using this script would look something like this:

.. code-block:: bash

    # set the environment variables
    export NSC_DATA_LIMIT=100000 # set data limit to 100,000 samples per dataset
    export NSC_EXPERIMENT_DIR=experiments # directory path where the experiment will be saved

    ## LOCAL training
    # start new training run using a config
    NSC_CONFIG=spell_checking/configs/sed_words.yaml spell_checking/scripts/train.sh

    # resume training from latest checkpoint of an experiment
    NSC_RESUME=<path_to_experiment_directory> spell_checking/scripts/train.sh

    ## SLURM training
    # starting distributed training on a SLURM cluster using sbatch
    # requires you to set the NSC_WORLD_SIZE environment variable (total number of GPUs used for training)
    # if you e.g. want to train on 4 nodes with 2 GPUs each set NSC_WORLD_SIZE=8
    NSC_CONFIG=spell_checking/configs/sed_words.yaml NSC_WORLD_SIZE=8 sbatch --nodes=4 --ntasks-per-node=2 --gres=gpu:2 spell_checking/scripts/train.sh

    # if you are in an interactive SLURM session (started e.g. with srun)
    # you probably want to train as if you are running locally, set NSC_FORCE_LOCAL=true and
    # start training without sbatch
    NSC_FORCE_LOCAL=true NSC_CONFIG=spell_checking/configs/sed_words.yaml spell_checking/scripts/train.sh

To retrain the models of this project see the ``train_slurm_<task>.sh`` scripts in this directory_ which were used for training all models.
These scripts do nothing more than setting some environment variables and calling the `train.sh`_  script.

.. note::

    Using the ``train_slurm_<task>.sh`` scripts for training is only possible on a SLURM cluster
    since they call the ``train.sh`` script using SLURMs sbatch command.

Reproduce
---------

We make all models that are needed to reproduce the results on the projects' benchmarks available as
`pretrained models <#pretrained-models>`_.
All pretrained models can be accessed either through the command line interface (``nsec``, ``nsed``, ``ntr``)
or the Python API.

The benchmarks can be found here: ``/nfs/students/sebastian-walter/masters_thesis/benchmarks``.
Every benchmark follows the same directory structure:

- <task>/<benchmark_group>/<benchmark_split>/corrupt.txt
- <task>/<benchmark_group>/<benchmark_split>/correct.txt

Here corrupt.txt is the input containing misspelled text and correct.txt is the groundtruth output. We
provide a `evaluation script`_ that can be used to evaluate model predictions on a given benchmark.

As an example, lets look at the steps that are necessary to evaluate our gnn+ model for word level spelling error detection on
the wikidump realistic benchmark using the command line interface:

1. Run model on benchmark:

.. code-block:: bash

   nsed -m "sed words:gnn+" \  # choose the model
   -f /nfs/students/sebastian-walter/masters_thesis/benchmarks/sed_words/wikidump/realistic/corrupt.txt \  # input file
   -o gnn_plus_predictions.txt  # save output to file

2. Evaluate model predictions:

.. code-block:: bash

   python spell_checking/benchmarks/scripts/evaluate.py \
   sed_words \  # benchmark type
   /nfs/students/sebastian-walter/masters_thesis/benchmarks/sed_words/wikidump/realistic/corrupt.txt \  # input file
   /nfs/students/sebastian-walter/masters_thesis/benchmarks/sed_words/wikidump/realistic/correct.txt \  # groundtruth file
   gnn_plus_predictions.txt  # predicted file

.. hint::

    By default a pretrained model is downloaded as a zip file and then extracted when you first use it. Since some models
    are quite large this can take some time. To cut this time all pretrained models can also be found as zip files in the directory
    ``/nfs/students/sebastian-walter/masters_thesis/zipped``. If you set the env variable
    ``NSC_DOWNLOAD_DIR`` to this directory, the models are loaded from this directory and must not be downloaded first.
    If you are running this project using Docker you can mount the directory to the containers download directory
    by specifying an additional volume flag:
    ``make DOCKER_ARGS="-v /nfs/students/sebastian-walter/masters_thesis/zipped:/nsc_download" run_docker_cpu``.

.. hint::

    To access the benchmarks if you are running this project with Docker you can mount the benchmark directory
    inside the Docker container using
    ``make DOCKER_ARGS="-v /nfs/students/sebastian-walter/masters_thesis/benchmarks:/benchmarks" run_docker_cpu``.
    The Docker container also provides additional commands for evaluating benchmarks that are basically
    wrappers around the `evaluation script`_ mentioned above.

.. _here: https://github.com/bastiscode/spell_check/tree/main/spell_checking/configs/train
.. _config for this task: https://github.com/bastiscode/spell_check/tree/main/spell_checking/configs/train/sed_words.yaml
.. _torchrun: https://pytorch.org/docs/stable/elastic/run.html
.. _train.sh: https://github.com/bastiscode/spell_check/tree/main/spell_checking/scripts/train.sh
.. _directory: https://github.com/bastiscode/spell_check/tree/main/spell_checking/scripts
.. _SLURM: https://slurm.schedmd.com/documentation.html
.. _evaluation script: https://github.com/bastiscode/spell_check/blob/main/spell_checking/benchmarks/scripts/evaluate.py
