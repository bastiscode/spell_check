.PHONY: create_benchmarks
create_benchmarks: create_neuspell_benchmarks create_own_benchmarks

.PHONY: create_neuspell_benchmarks
create_neuspell_benchmarks:
	@python scripts/create_neuspell_benchmarks.py

# determines benchmark split: one of {dev, test}
SPLIT := test

.PHONY: create_own_benchmarks
create_own_benchmarks:
	bash ./scripts/create_benchmarks.sh

.PHONY: evaluate_sed_sequence
evaluate_sed_sequence:
	@python scripts/evaluate.py \
	--groundtruths $(SPLIT)/sed_sequence/*/*/correct.txt \
	--predictions $(SPLIT)/sed_sequence/results/*/*/*.txt \
	--inputs $(SPLIT)/sed_sequence/*/*/corrupt.txt \
	--metrics binary_f1 binary_precision binary_recall sequence_accuracy

.PHONY: evaluate_sed_words
evaluate_sed_words:
	@python scripts/evaluate.py \
	--groundtruths $(SPLIT)/sed_words/*/*/correct.txt \
	--predictions $(SPLIT)/sed_words/results/*/*/*.txt \
	--inputs $(SPLIT)/sed_words/*/*/corrupt.txt \
	--metrics binary_f1 binary_precision binary_recall sequence_accuracy word_accuracy

.PHONY: evaluate_sec
evaluate_sec:
	@python scripts/evaluate.py \
	--groundtruths $(SPLIT)/sec/*/*/correct.txt \
	--predictions $(SPLIT)/sec/results/*/*/*.txt \
	--inputs $(SPLIT)/sec/*/*/corrupt.txt \
	--metrics sequence_accuracy f1 precision recall med mned

#.PHONY: evaluate_tokenization_repair
#evaluate_tokenization_repair:
#	@python scripts/evaluate.py \
#	--groundtruths $(SPLIT)/tokenization_repair/*/$(SPLIT)/correct.txt \
#	--predictions $(SPLIT)/tokenization_repair/results/*/$(SPLIT)/*.txt \
#	--inputs $(SPLIT)/tokenization_repair/*/*/corrupt.txt \
#	--metrics sequence_accuracy f1 precision recall med mned

.PHONY: evaluate_all
evaluate_all: evaluate_sed_words evaluate_sed_sequence evaluate_sec evaluate_tokenization_repair

.PHONY: run_experiments_on_benchmarks
run_experiments_on_benchmarks:
	bash ./scripts/run_experiments_on_benchmarks.sh

.PHONY: run_baselines_on_benchmarks
run_baselines_on_benchmarks:
	bash ./scripts/run_baselines_on_benchmarks.sh
