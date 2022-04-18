#!/bin/bash

echo """

Welcome to the Neural Spell Checking (NSC) project!

You can now use the NSC command line tools, try out the following commands for each task:
  Tokenization repair
    ntr -r \"Tihsis a sente nse with incor ect whitespac ign.\"

  Spelling error detection
    nsed -d \"This sentense has an eror!\"

  Spelling error correction
    nsec -c \"This sentense has an eror!\"

You can do many more things using these three commands. Execute ntr -h, nsed -h or nsec -h to see all available options.

Show the project documentation as man page execute:
  make show_docs

Show this help message:
  make help

	"""
