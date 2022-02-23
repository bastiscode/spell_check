.PHONY: components
components:
	cd components/plotting/streamlit_network_graph/frontend && npm run build
	cd components/plotting && rm -rf build && rm -rf dist && python setup.py sdist bdist_wheel

.PHONY: install
install:
	cd components/plotting && pip install --force-reinstall dist/streamlit_network_graph-*-py3-none-any.whl
