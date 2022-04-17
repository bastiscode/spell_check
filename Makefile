.PHONY: install
install:
	@echo "Installing nsc library"
	pip install .[train]

.PHONY: docker
docker: build_docker run_docker

.PHONY: build_docker
build_docker:
	@echo "Building Dockerfile"
	docker build -t nsc .

.PHONY: run_docker
run_docker:
	@echo "Running Dockerfile"
	docker run -it --gpus all nsc

.PHONY: docs
docs:
	@echo "Building docs"
	cd docs && sphinx-apidoc -f -o . ../nsc && make html

.PHONY: checkstyle
checkstyle:
	@echo "Running pep8 (pycodestyle)\n-----------------"
	pycodestyle --exclude="data,spell_checking,third_party" --max-line-length=120 .

.PHONY: tests
tests:
	pytest -s tests -n auto --disable-pytest-warnings
