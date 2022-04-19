Reproduce results
================================

To reproduce the results you will need the training data. Everything
you need (preprocessed training data, tokenizer, dictionaries, etc.) can be found
under ``/nfs/students/sebastian-walter/masters_thesis/data``.

You also need to set the following two special environment variables.

.. note::
    Of course you can also copy the training data folder to every other
    place you like and adjust ``NSC_DATA_DIR`` accordingly.

.. code-block:: bash

    # set the data directory
    export NSC_DATA_DIR=/nfs/students/sebastian-walter/masters_thesis/data

    # set the config directory (necessary to be able to compose
    # the final training config from sub-configs)
    export NSC_CONFIG_DIR=spell_checking/configs

After that all you train your own models using a training config.
For all available training configs see the ``configs/train`` directory.


