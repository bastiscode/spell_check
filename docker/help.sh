#!/bin/bash

echo """

Welcome to the Neural Spell Checking (NSC) project!

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

Show the project documentation as man page:
  make show_docs

Show this help message:
  make help

	"""
