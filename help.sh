#!/bin/bash

echo """

Welcome to the Neural Spell Checking (NSC) project!
We describe some things you can do with this project below.

You can now use the NSC command line tools. As a starting point, try out the following commands for each task:
  Tokenization repair
    ntr -r \"Tihsis a sente nse with incor ect whitespac ign.\"

  Spelling error detection (word level)
    nsed -d \"This sentense has an eror!\"

  Spelling error correction
    nsec -c \"This sentense has an eror!\"

You can do many more things using these three command line tools. Execute ntr -h, nsed -h or nsec -h to see all available options.
You can also evaluate the predictions of a model run with one of the ntr, nsed or nsec commands. To do that, use
the following commands and specify the paths of the input, groundtruth and prediction file you want to evaluate:

  Tokenization repair
    evaluate_tr <input_file> <groundtruth_file> <prediction_file>

  Spelling error detection (sequence level)
    evaluate_seds <input_file> <groundtruth_file> <prediction_file>

  Spelling error detection (word level)
    evaluate_sedw <input_file> <groundtruth_file> <prediction_file>

  Spelling error correction
    evaluate_sec <input_file> <groundtruth_file> <prediction_file>

You can start a spell checking server that gives you access to a JSON API (see server_config/server.yaml for configuration options):

  nserver -c server_config/server.yaml

You can host a spell checking web app on port 8080 which uses the JSON API started by nserver as backend
(publish port 8080 when running this image to access the webapp from outside the container):

  python -m http.server --directory webapp/build/web 8080 & nserver -c server_config/server.yaml && fg

  Note: The webapp currently does not provide the full feature set of the command line tools or Python API
  (e.g. using beam search for spelling correction or converting outputs between tasks). However, these are special cases
  and for over 95% of use cases the webapp is sufficient and provides a much better user experience than the command line tools.

We provide quick-start make targets for running the spell checking server and running the spell checking webapp
(quit this container and execute one of the following two commands):

  Start the spell checking server:
    make run_docker_server

  Start the spell checking web app:
    make run_docker_webapp

Show the project documentation as man page:
  make show_docs

Show this help message:
  make help

	"""
