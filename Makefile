.PHONY: install
install:
	@echo "Installing nsc library and other relevant software"
	@bash nsc/scripts/setup_rust_extensions.sh
	@pip install .[train,index,docs,test] -f https://data.dgl.ai/wheels/repo.html
	@python -m spacy download en_core_web_lg

# use this to change the docker cmd to e.g. wharfer
DOCKER_CMD ?= docker
# use this to pass additional flags to the docker command
# we set '-e NSC_DUMMY=dummy' here as a placeholder argument
# to not get invalid reference format errors from docker
DOCKER_ARGS ?= -e NSC_DUMMY=dummy

.PHONY: build_docker
build_docker:
	@echo "Building nsc image"
	@$(DOCKER_CMD) build -t nsc .

.PHONY: run_docker
run_docker:
	@echo "Running nsc image"
	@$(DOCKER_CMD) run $(DOCKER_ARGS) -it --name nsc nsc

.PHONY: run_docker_server
run_docker_server:
	@echo "Running nsc server"
	@$(DOCKER_CMD) run $(DOCKER_ARGS) -it --name nsc_server --entrypoint bash nsc \
 	-c "nserver -c server_config/server.yaml"

.PHONY: run_docker_webapp
run_docker_webapp:
	@echo "Running nsc webapp"
	@$(DOCKER_CMD) run $(DOCKER_ARGS) -it --name nsc_webapp --entrypoint bash -p 44444:44444 nsc \
	-c "python -m http.server --directory webapp/build/web 8080 & nserver -c server_config/server.yaml && fg"

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

# for docker image
.PHONY: help
help:
	@bash help.sh

.PHONY: show_docs
show_docs: docs
	@man docs/nsc_man
