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
	@rm -f $(SPLIT)/sed_sequence_tables/results.json
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sed_sequence \
	--benchmark-type sed_sequence \
	--result-dir $(SPLIT)/sed_sequence/results \
	--save-dir $(SPLIT)/sed_sequence_tables

	@glow -w 150 $(SPLIT)/sed_sequence_tables/*.md

	cd ../../../latex && pdflatex -synctex=1 -interaction=batchmode thesis_main.tex

.PHONY: evaluate_sed_words
evaluate_sed_words:
	@rm -f $(SPLIT)/sed_words_tables/results.json
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sed_words \
	--benchmark-type sed_words \
	--result-dir $(SPLIT)/sed_words/results \
	--save-dir $(SPLIT)/sed_words_tables \
	--dictionary ../../data/dictionaries/merged_train_100k.txt

	@glow -w 150 $(SPLIT)/sed_words_tables/*.md

	@rm -f $(SPLIT)/sed_words_advanced_tables/results.json
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sed_words \
	--benchmark-type sed_words_advanced \
	--result-dir $(SPLIT)/sed_words/results \
	--save-dir $(SPLIT)/sed_words_advanced_tables \
	--dictionary ../../data/dictionaries/merged_train_100k.txt

	@glow -w 150 $(SPLIT)/sed_words_advanced_tables/*.md

	cd ../../../latex && pdflatex -synctex=1 -interaction=batchmode thesis_main.tex

.PHONY: evaluate_sec
evaluate_sec:
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sec \
	--benchmark-type sec \
	--result-dir $(SPLIT)/sec/results \
	--save-dir $(SPLIT)/sec_tables

	@glow -w 150 $(SPLIT)/sec_tables/*.md

	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sec \
	--benchmark-type sec_advanced \
	--result-dir $(SPLIT)/sec/results \
	--save-dir $(SPLIT)/sec_advanced_tables

	@glow -w 150 $(SPLIT)/sec_advanced_tables/*.md

	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sec \
	--benchmark-type sec_whitespace \
	--result-dir $(SPLIT)/sec/results \
	--save-dir $(SPLIT)/sec_whitespace_tables

	@glow -w 150 $(SPLIT)/sec_whitespace_tables/*.md

	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/sec \
	--benchmark-type sec_spell_checking \
	--result-dir $(SPLIT)/sec/results \
	--save-dir $(SPLIT)/sec_spell_checking_tables

	@glow -w 150 $(SPLIT)/sec_spell_checking_tables/*.md

	cd ../../../latex && pdflatex -synctex=1 -interaction=batchmode thesis_main.tex

.PHONY: evaluate_tokenization_repair
evaluate_tokenization_repair:
	@python scripts/evaluate_thesis.py \
	--benchmark-dir $(SPLIT)/tokenization_repair \
	--benchmark-type tokenization_repair \
	--result-dir $(SPLIT)/tokenization_repair/results \
	--save-dir $(SPLIT)/tokenization_repair_tables

	@glow -w 150 $(SPLIT)/tokenization_repair_tables/*.md

.PHONY: evaluate_benchmark_statistics
evaluate_benchmark_statistics:
	@python scripts/generate_statistics_thesis.py \
	--benchmark-dir $(SPLIT)/sed_sequence \
	--benchmark-type sed_sequence \
	--dictionary ../../data/dictionaries/merged_train_100k.txt \
	--out-dir $(SPLIT)/benchmark_statistics_tables/sed_sequence_benchmark_statistics

	@python scripts/generate_statistics_thesis.py \
	--benchmark-dir $(SPLIT)/sec \
	--benchmark-type sed_words_and_sec \
	--dictionary ../../data/dictionaries/merged_train_100k.txt \
	--out-dir $(SPLIT)/benchmark_statistics_tables/sec_benchmark_statistics

	cd ../../../latex && pdflatex -synctex=1 -interaction=batchmode thesis_main.tex

.PHONY: evaluate_all
evaluate_all: evaluate_benchmark_statistics evaluate_sed_words evaluate_sed_sequence evaluate_sec evaluate_tokenization_repair

.PHONY: run_experiments_on_benchmarks
run_experiments_on_benchmarks:
	bash ./scripts/run_experiments_on_benchmarks.sh

.PHONY: run_baselines_on_benchmarks
run_baselines_on_benchmarks:
	bash ./scripts/run_baselines_on_benchmarks.sh
