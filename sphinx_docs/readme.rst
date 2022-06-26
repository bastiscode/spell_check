Neural spell checking using Transformers and Graph Neural Networks
==================================================================

This project is about detecting and correcting spelling errors using Transformers and
Graph Neural Networks. Visit the `documentation`_ (which also includes this README)
for more information on how to reproduce results and train your own models.

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

There are three main ways to use this project:

1. Command line
2. Python API
3. Web app

We recommend that you start out with the `Web app`_ using the Docker_ setup
which provides the best user experience overall.

Command line interfaces
~~~~~~~~~~~~~~~~~~~~~~~

After installation there will be four commands available in your environment:

1. ``nsec`` for neural spelling error correction
2. ``nsed`` for neural spelling error detection
3. ``ntr`` for neural tokenization repair
4. ``nserver`` for running a spell checking server

By default the first three commands take input from `stdin`, run their respective task on the
input line by line and print their output line by line to `stdout`. The fourth command allows you
to start a spell checking server that gives you access to all models via a JSON API.
It is e.g. used as backend for the `Web app`_. The server can be configured
using a yaml file (see `server_config/server.yaml`_ for the configuration options).

Let's look at some basic examples on how to use the command line tools.

**Spelling error correction using** ``nsec``

.. code-block:: bash

    # correct text by piping into nsec,
    echo "This is an incorect sentense!" | nsec
    cat path/to/file.txt | nsec

    # by using the -c flag,
    nsec -c "This is an incorect sentense!"

    # by passing a file,
    nsec -f path/to/file.txt

    # or by starting an interactive session
    nsec -i

    # you can also use spelling correction to detect errors on word and sequence level
    nsec -c "This is an incorect sentense!" --convert-output word_detections
    nsec -c "This is an incorect sentense!" --convert-output sequence_detections

**Spelling error detection using** ``nsed``

.. code-block:: bash

    # detect errors by piping into nsed,
    echo "This is an incorect sentense!" | nsed
    cat path/to/file.txt | nsed

    # by using the -d flag,
    nsed -d "This is an incorect sentense!"

    # by passing a file,
    nsed -f path/to/file.txt

    # or by starting an interactive session
    nsed -i

    # you can also use word level spelling error detections to detect sequence level errors
    nsed -d "This is an incorect sentense!" --convert-sequence

**Tokenization repair using** ``ntr``

.. code-block:: bash

    # repair text by piping into ntr,
    echo "Thisis an inc orect sentens e!" | ntr
    cat path/to/file.txt | ntr

    # by using the -r flag,
    ntr -r "Thisis an inc orect sentens e!"

    # by passing a file,
    ntr -f path/to/file.txt

    # or by starting an interactive session
    ntr -i

You can also combine the ``ntr``, ``nsed``, and ``nsec`` commands in a variety of ways.
Some examples are shown below.

.. code-block:: bash

    # repair and detect
    echo "Repi arand core ct tihs sen tens!" | ntr | nsed
    # to view both the repaired text and the detections use
    echo "Repi arand core ct tihs sen tens!" | ntr | nsed --sec-out

    # repair and correct
    echo "Repi arand core ct tihs sen tens!" | ntr | nsec

    # repair and correct a file and save the output
    ntr -f path/to/file.txt | nsec --progress -o path/to/output_file.txt

    # repair, detect and correct
    # (this pipeline uses the spelling error detection output
    # to guide the spelling error correction model to correct only the misspelled words)
    echo "Repi arand core ct tihs sen tens!" | ntr | nsed --sec-out | nsec --sed-in

    # some detection and correction models (e.g. tokenization repair+, tokenization repair++, transformer with tokenization repair nmt)
    # can natively deal with incorrect whitespacing in text, so there is no need to use ntr before them if you want to process
    # text with whitespacing errors
    nsed -d "core ct thissen tense!" -m "sed words:tokenization repair+" --sec-out
    nsec -c "core ct thissen tense!" -m "tokenization repair++"
    nsec -c "core ct thissen tense!" -m "transformer with tokenization repair nmt"

There are a few other command line options available for the ``nsec``, ``nsed`` and ``ntr`` commands. Inspect
them by passing the ``-h / --help`` flag to the commands.

**Running a spell checking server using** ``nserver``

.. code-block:: bash

    # start the spell checking server, we provide a default config at server_config/server.yaml
    nserver -c path/to/config.yaml

Python API
~~~~~~~~~~

We also provide a Python API for you to use spell checking models directly in code. Below are basic
code examples on how to use the API. For the full documentation of all classes, methods, etc. provided by
the Python API see the `nsc package documentation <#module-nsc>`_.

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

Web app
~~~~~~~

.. image:: https://github.com/bastiscode/spell_check/raw/main/images/web_app.png
    :alt: Web app screenshot

The web app source can be found under `webapp/build/web`. You can host it e.g. using Pythons
built-in http server:

.. code-block:: bash

    python -m http.server -d directory webapp/build/web 8080

The web app expects a spell checking server running under the same hostname on port 44444 (we are currently looking
into ways to be able to set the hostname and port for the server endpoint at runtime). If you are running the web app
locally you can simply start the spell checking server using ``nserver``:

.. code-block:: bash

    nserver -c server_config/server.yaml

Then go to your browser to port 8080 to use the web app.

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

    make run_docker

You can also pass additional Docker arguments to the make commands by specifying ``DOCKER_ARGS``. For example,
to be able to access your GPUs inside the container run ``make run_docker DOCKER_ARGS="--gpus all"``. As another example,
to mount an additional directory inside the container use
``make run_docker DOCKER_ARGS="-v /path/to/outside/directory:/path/to/container/directory"``.

We provide two convenience commands for starting a spell checking server or hosting the web app:

Start a spell checking server on port <outside_port>:

.. code-block:: bash

    make run_docker_server DOCKER_ARGS="-p <outside_port>:44444"

Start the web app on port <outside_port> (also runs the spell checking server):

.. code-block:: bash

    make run_docker_webapp DOCKER_ARGS="-p <outside_port>:8080"

.. hint::
    If you build the Docker image on an AD Server you probably want to use wharfer instead of
    Docker. To do that call the make commands with the additional argument ``DOCKER_CMD=wharfer``,
    e.g. ``make build_docker DOCKER_CMD=wharfer``.

.. note::
    The Docker setup is only intended to be used for running the command line tools, Python API, or web app
    with pretrained or your own models and evaluating benchmarks, but not for training.

.. note::
    Running the Docker container with GPU support assumes that you have the `NVIDIA Container Toolkit`_ installed.

.. _NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
.. _documentation: https://bastiscode.github.io/spell_check
.. _server_config/server.yaml: https://github.com/bastiscode/spell_check/tree/main/server_config/server.yaml
