.PHONY: create_benchmarks
create_benchmarks: create_neuspell_benchmarks create_own_benchmarks

.PHONY: create_neuspell_benchmarks
create_neuspell_benchmarks:
	@python scripts/create_neuspell_benchmarks.py \
	--neuspell-dir ../../data/raw/neuspell/traintest \
	--benchmark-dir `pwd`/test \
	--benchmarks bea60k \
	--seed 22

# determines benchmark split: one of {dev, test}
SPLIT := test

.PHONY: create_own_benchmarks
create_own_benchmarks:
	bash ./scripts/create_benchmarks.sh

.PHONY: evaluate_sed_sequence
evaluate_sed_sequence:
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sed_sequence \
	--benchmark-type sed_sequence \
	--result-dir $(SPLIT)/sed_sequence/results \
	--save-dir $(SPLIT)/sed_sequence_tables

	@glow -w 150 $(SPLIT)/sed_sequence_tables/*.md

.PHONY: evaluate_sed_words
evaluate_sed_words:
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sed_words \
	--benchmark-type sed_words \
	--result-dir $(SPLIT)/sed_words/results \
	--save-dir $(SPLIT)/sed_words_tables

	@glow -w 150 $(SPLIT)/sed_words_tables/*.md

.PHONY: evaluate_sec
evaluate_sec:
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sec \
	--benchmark-type sec \
	--result-dir $(SPLIT)/sec/results \
	--save-dir $(SPLIT)/sec_tables

	@glow -w 150 $(SPLIT)/sec_tables/*.md

.PHONY: evaluate_tokenization_repair
evaluate_tokenization_repair:
	echo "to be implemented"

.PHONY: evaluate_all
evaluate_all: evaluate_sed_words evaluate_sed_sequence evaluate_sec evaluate_tokenization_repair

.PHONY: run_experiments_on_benchmarks
run_experiments_on_benchmarks:
	bash ./scripts/run_experiments_on_benchmarks.sh

.PHONY: run_baselines_on_benchmarks
run_baselines_on_benchmarks:
	bash ./scripts/run_baselines_on_benchmarks.sh
