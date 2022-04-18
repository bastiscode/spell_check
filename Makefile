.PHONY: install
install:
	@echo "Installing nsc library and other relevant software"
	@pip install .[train,index] -f https://data.dgl.ai/wheels/repo.html
	@python -m spacy download en_core_web_lg
	@pip install -r requirements.txt

.PHONY: build_docker
build_docker:
	@echo "Building Docker image"
	@docker build -t nsc .

.PHONY: run_docker_gpu
run_docker_gpu:
	@echo "Running nsc image with GPU support"
	@docker run -it --gpus all --name nsc_gpu nsc

.PHONY: run_docker_cpu
run_docker_cpu:
	@echo "Running nsc image on CPU"
	@docker run -it --name nsc_cpu nsc

.PHONY: docs
docs:
	@echo "Building docs"
	@cp README.rst docs/readme.rst
	@sphinx-apidoc -M -f -o docs nsc nsc/train.py nsc/version.py nsc/preprocess.py nsc/scripts nsc/utils \
	nsc/tasks nsc/data nsc/models nsc/modules nsc/api
	@make -C docs singlehtml
	@make -C docs man

.PHONY: checkstyle
checkstyle:
	@echo "Running pep8 (pycodestyle)\n-----------------"
	@pycodestyle --exclude="data,spell_checking,third_party" --max-line-length=120 .

.PHONY: tests
tests:
	@pytest -s tests -n auto --disable-pytest-warnings
