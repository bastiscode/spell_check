.PHONY: setup
setup:
	@pip install maturin==0.12.12

.PHONY: install
install:
	@maturin build --release
	@pip install .
