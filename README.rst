Neural spell checking using Transformers and Graph Neural Networks
==================================================================

This project is about detecting and correcting spelling errors using Transformers and
Graph Neural Networks. Visit the `documentation`_ (which also includes this README)
for information on how to reproduce results and train your own models
as well as a more detailed description of the ``nsc`` `Python API`_.

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

**Spelling error correction using** ``nsec``

.. code-block:: bash

    # correct text by piping into nsec,
    echo "This is an incorect sentense!" | nsec
    cat path/to/file.txt | nsec

    # by using the -c flag
    nsec -c "This is an incorect sentense!"

    # or by passing a file
    nsec -f path/to/file.txt

**Spelling error detection using** ``nsed``

.. code-block:: bash

    # detect errors by piping into nsed,
    echo "This is an incorect sentense!" | nsed
    cat path/to/file.txt | nsed

    # by using the -d flag
    nsed -d "This is an incorect sentense!"

    # or by passing a file
    nsed -f path/to/file.txt

**Tokenization repair using** ``ntr``

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
code examples on how to use the API.

**Spelling error correction**

.. code-block:: python

    from nsc import SpellingErrorCorrector, get_available_spelling_error_correction_models

    # show all spelling error correction models
    print(get_available_spelling_error_correction_models())

    # use a pretrained model
    sec = SpellingErrorCorrector.from_pretrained()
    # correct errors in text
    correction = sec.correct_text("Tihs text has erors!")
    print(correction)
    # correct errors in file
    corrections = sec.correct_file("path/to/file.txt")
    print(correction)

**Spelling error detection**

.. code-block:: python

    from nsc import SpellingErrorDetector, get_available_spelling_error_detection_models

    # show all spelling error detection models
    print(get_available_spelling_error_detection_models())

    # use a pretrained model
    sed = SpellingErrorDetector.from_pretrained()
    # detect errors in text
    detection = sed.detect_text("Tihs text has erors!")
    print(detection)
    # detect errors in file
    detections = sed.detect_file("path/to/file.txt")
    print(detections)

**Tokenization repair**

.. code-block:: python

    from nsc import TokenizationRepairer, get_available_tokenization_repair_models

    # show all tokenization repair models
    print(get_available_tokenization_repair_models())

    # use a pretrained model
    tr = TokenizationRepairer.from_pretrained()
    # repair tokenization in text
    repaired_text = tr.repair_text("Ti hstext h aserors!")
    print(repaired_text)
    # repair tokenization in file
    repaired_file = tr.repair_file("path/to/file.txt")
    print(repaired_file)

Docker
------

This project can also be run using Docker.
Inside the Docker container both the `Command line interfaces`_ and `Python API`_ are available for you to use.
You can also evaluate model predictions on benchmarks.

Build the Docker image:

.. code-block:: bash

    make build_docker

Start a Docker container:

.. code-block:: bash

    # run the docker container with GPU support
    make run_docker_gpu
    # or with CPU support only
    make run_docker_cpu

You can also pass additional Docker arguments to the make commands by specifying ``DOCKER_ARGS``. For example,
to mount an additional directory inside the container use
``make DOCKER_ARGS=" -v /path/to/outside/directory:/path/to/container/directory" run_docker_gpu``
(notice the leading whitespace!).

.. hint::

    If you build the Docker image on an AD Server you probably want to use wharfer instead of
    Docker. To do that call the make commands with the additional argument ``DOCKER_CMD=wharfer``,
    e.g. ``make DOCKER_CMD=wharfer build_docker``.

.. note::
    The Docker setup is only intended to be used for running the command line tools/Python API with pretrained or
    your own models and evaluating benchmarks, but not for training.

.. note::
    Running the Docker container with GPU support assumes that you have the `NVIDIA Container Toolkit`_ installed.

.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
.. _documentation: https://bastiscode.github.io/spell_check