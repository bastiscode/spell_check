.PHONY: install
install:
	@echo "Installing nsc library and other relevant software"
	@bash nsc/scripts/setup_rust_extensions.sh
	@pip install .[train,index,docs,test] -f https://data.dgl.ai/wheels/repo.html
	@python -m spacy download en_core_web_lg

DOCKER_CMD ?= "docker"
DOCKER_ARGS ?= ""

.PHONY: build_docker
build_docker:
	@echo "Building Docker image"
	@$(DOCKER_CMD) build -t nsc .

.PHONY: run_docker_gpu
run_docker_gpu:
	@echo "Running nsc image with GPU support"
	@$(DOCKER_CMD) run $(DOCKER_ARGS) -e NSC_CACHE_DIR=/nsc_cache -e NSC_DOWNLOAD_DIR=/nsc_download \
	--rm -it --gpus all --name nsc_gpu nsc

.PHONY: run_docker_cpu
run_docker_cpu:
	@echo "Running nsc image on CPU"
	@$(DOCKER_CMD) run $(DOCKER_ARGS) -e NSC_CACHE_DIR=/nsc_cache -e NSC_DOWNLOAD_DIR=/nsc_download \
	--rm -it --name nsc_cpu nsc

.PHONY: docs
docs:
	@echo "Building docs"
	@cp README.rst sphinx_docs/readme.rst
	@sphinx-apidoc -M -f -o sphinx_docs nsc nsc/train.py nsc/version.py nsc/preprocess.py nsc/scripts nsc/utils \
	nsc/tasks nsc/data nsc/models nsc/modules nsc/api
	@make -C sphinx_docs singlehtml
	@make -C sphinx_docs man
	@mkdir -p docs
	@cp -a sphinx_docs/_build/singlehtml/. docs
	@cp -a sphinx_docs/_build/man/nsc.1 docs/nsc_man

.PHONY: checkstyle
checkstyle:
	@echo "Running pep8 (pycodestyle)"
	@pycodestyle --exclude="data,spell_checking,third_party" --max-line-length=120 .

.PHONY: tests
tests:
	@echo "Running pytest"
	@pytest -s tests -n auto --disable-pytest-warnings
