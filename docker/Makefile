.PHONY: help
help:
	@bash help.sh

.PHONY: show_docs
show_docs:
	@make -C .. docs
	@man ../docs/nsc_man

.PHONY: webapp
webapp:
	$(shell python -m http.server --directory ../web_app/build/web 8080 & ../bin/nserver -c ../server_config/server.yaml && fg)
