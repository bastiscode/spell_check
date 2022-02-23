.PHONY: tensorboard
tensorboard:
	tensorboard --logdir experiments

.PHONY: checkstyle
checkstyle:
	@echo "Running pep8 (pycodestyle)\n-----------------"
	pycodestyle --exclude="build,data,third_party,norvig" --max-line-length=120 .
	@echo "\nRunning mypy\n-----------------"
	mypy .

.PHONY: python_tests
python_tests:
	pytest -s tests -n auto --disable-pytest-warnings

.PHONY: other_tests
other_tests:
	accelerate test --config accelerate_config.json

.PHONY: tests
tests: python_tests other_tests

# some helpers for the KI Cluster at Uni Freiburg

.PHONY: interactive
interactive:
	srun --time=12:00:00 --gres=gpu:1 --partition=alldlc_gpu-rtx2080 --pty bash -i

.PHONY: login
login:
	ssh -t swalter@kislogin2.rz.ki.privat \
	"cd /work/dlclarge1/swalter-masters_thesis/masters_thesis && \
	export DGLBACKEND=pytorch && \
	export PYTHONPATH=$$PYTHONPATH:/work/dlclarge1/swalter-masters_thesis/masters_thesis; bash --login"

.PHONY: sync
sync:
	rsync -ah --progress --exclude-from="rsync_exclude.txt" . \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis

.PHONY: tensorboard_remote
tensorboard_remote:
	ssh -N -L localhost:16006:localhost:$${PORT:-14555} swalter@kislogin2.rz.ki.privat

.PHONY: tensorboard_remote_server
tensorboard_remote_server:
	ssh -t swalter@kislogin2.rz.ki.privat \
	"cd /work/dlclarge1/swalter-masters_thesis/masters_thesis && \
	export DGLBACKEND=pytorch && \
	export PYTHONPATH=$$PYTHONPATH:/work/dlclarge1/swalter-masters_thesis/masters_thesis && \
	. ../env/bin/activate && tensorboard --logdir experiments --port 14555; bash --login"

EXP_TYPE := ""
EXP_REGEX := "*"
.PHONY: sync_experiments
sync_experiments:
	rsync -ah --compress --progress \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/experiments/$(EXP_TYPE)/$(EXP_REGEX) \
	--exclude "*checkpoints/checkpoint_[0-9]*.pt" \
	--exclude "*checkpoints/checkpoint_last.pt" \
	experiments/$(EXP_TYPE)

.PHONY: sync_tokenizers
sync_tokenizers:
	rsync -ah --progress \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/data/tokenizers/ data/tokenizers

.PHONY: sync_dictionaries
sync_dictionaries:
	rsync -ah --progress \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/data/dictionaries/ data/dictionaries

.PHONY: sync_spell_check_index
sync_spell_check_index:
	rsync -ah --progress \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/data/spell_check_index/ data/spell_check_index

.PHONY: sync_benchmarks
sync_benchmarks:
	mkdir -p spelling_correction/benchmarks/test
	rsync -ah --progress \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/spelling_correction/benchmarks/test/* \
	spelling_correction/benchmarks/test

	mkdir -p spelling_correction/benchmarks/dev
	rsync -ah --progress \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/spelling_correction/benchmarks/dev/* \
	spelling_correction/benchmarks/dev

.PHONY: sync_tensorboard
sync_tensorboard:
	rsync -ah --progress --relative \
	swalter@kislogin2.rz.ki.privat:/work/dlclarge1/swalter-masters_thesis/masters_thesis/experiments/./*/*/tensorboard \
	tensorboard

# other stuff

.PHONY: loc
loc:
	git ls-files | grep '\.py' | xargs wc -l

.PHONY: matthias-thesis
matthias-thesis:
	docker run --runtime=nvidia -it \
	-v $(shell pwd)/third_party/matthias-master/thesis:/external \
	-v $(shell pwd)/spelling_correction/benchmarks/test/sec:/sec \
	-p 1234:1234 matthias-hertel-thesis
