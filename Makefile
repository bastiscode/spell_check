.PHONY: install
install:
	@echo "Installing nsc library"
	pip install .[train] -f https://data.dgl.ai/wheels/repo.html

.PHONY: build_docker
build_docker:
	@echo "Building Dockerfile"
	docker build -t nsc .

.PHONY: run_docker_gpu
run_docker_gpu:
	@echo "Running nsc Dockerfile with GPU support"
	docker run -it --gpus all nsc

.PHONY: run_docker_cpu
run_docker_cpu:
	@echo "Running nsc Dockerfile on CPU"
	docker run -it nsc

.PHONY: docs
docs:
	@echo "Building docs"
	sphinx-apidoc -M -f -o docs nsc nsc/train.py nsc/version.py nsc/preprocess.py nsc/scripts nsc/utils \
	nsc/tasks nsc/data nsc/models nsc/modules nsc/api && make -C docs html

.PHONY: checkstyle
checkstyle:
	@echo "Running pep8 (pycodestyle)\n-----------------"
	pycodestyle --exclude="data,spell_checking,third_party" --max-line-length=120 .

.PHONY: tests
tests:
	pytest -s tests -n auto --disable-pytest-warnings
