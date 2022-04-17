Neural spell checking using Transformers and Graph Neural Networks
========================================================================

This project is about detecting and correcting spelling errors using Transformers and
Graph Neural Networks.

Installation
------------

Clone the repository

.. code-block:: bash

    git clone git@github.com:bastiscode/spell_check.git

Install from source (alternatively you can use Docker_)

.. code-block:: bash

    make install

Usage
-----

There are two main ways to use this project.
Either via the command line or by directly using the Python API.

Command line interfaces
~~~~~~~~~~~~~~~~~~~~~~~

After installation there will be three commands available in your environment:

    1. ``nsec`` for neural spelling error correction
    2. ``nsed`` for neural spelling error detection
    3. ``ntr`` for neural tokenization repair

By default all three commands take input from `stdin`, run their respective task on the
input line by line and print their output line by line to `stdout`.

Spelling error correction using **nsec**

.. code-block:: bash

    # correct text by piping into nsec,
    echo "This is an incorect sentense!" | nsec
    cat path/to/file.txt | nsec

    # by using the -c flag
    nsec -c "This is an incorect sentense!"

    # or by passing a file
    nsec -f path/to/file.txt

Spelling error detection using **nsed**

.. code-block:: bash

    # detect errors by piping into nsed,
    echo "This is an incorect sentense!" | nsed
    cat path/to/file.txt | nsed

    # by using the -d flag
    nsed -d "This is an incorect sentense!"

    # or by passing a file
    nsed -f path/to/file.txt

Tokenization repair using **ntr**

.. code-block:: bash

    # repair text by piping into ntr,
    echo "Thisis an inc orect sentens e!" | ntr
    cat path/to/file.txt | ntr

    # by using the -r flag
    ntr -r "Thisis an inc orect sentens e!"

    # or by passing a file
    ntr -f path/to/file.txt

You can also combine the ``ntr``, ``nsed``, and ``nsec`` commands in a variety of ways.
Some examples are shown below.

.. code-block:: bash

    # repair and detect
    echo "Repi arand detec erors in tihssen tence!" | ntr | nsed

    # repair and correct
    echo "Repi arand core ct tihssen tens!" | ntr | nsec

    # repair, detect and correct
    # (this pipeline uses the spelling error detection output
    # to guide the spelling error correction model to correct only the misspelled words)
    echo "Repi arand core ct tihssen tens!" | ntr | nsed --sec-out | nsec --sed-in

    # repair and correct a file and save the output
    ntr -f path/to/file.txt | nsec --progress -o path/to/output_file.txt

There are a few other command line options available for the ``nsec``, ``nsed`` and ``ntr`` commands. Inspect
them by passing the ``-h / --help`` flag to the commands.

Python API
~~~~~~~~~~

We also provide a Python API for you to use spell checking models directly in code. Below are basic
code examples on how to use the API. For a full documentation on all provided classes, methods and arguments
see the ``nsc`` Python API documentation.

Spelling error correction

.. literalinclude:: examples/sec.py
    :language: python

Spelling error detection

.. literalinclude:: examples/sed.py
    :language: python

Tokenization repair

.. literalinclude:: examples/tokenization_repair.py
    :language: python

Docker
------

This project can also be run using Docker.
Inside the Docker container both the `Command line interfaces`_ and `Python API`_ are available for you to use.


To build the Docker image

.. code-block:: bash

    make build_docker

To start a Docker container

.. code-block:: bash

    # run the docker container with GPU support
    make run_docker_gpu
    # or with CPU support
    make run_docker_cpu

.. note::
    Running the Docker container with GPU support assumes that you have the `NVIDIA Container Toolkit`_ installed.

.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
