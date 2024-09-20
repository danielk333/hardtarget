# Minimal makefile for Sphinx documentation

# html
SPHINX_OPTS    ?=
SPHINX_BUILD    = sphinx-build
DOCS_SOURCE     = source
DOCS_BUILD      = build

# apidoc
SPHINX_APIDOC   = sphinx-apidoc
APIDOCS_SOURCE   = ../src/hardtarget
APIDOCS_BUILD    = $(DOCS_SOURCE)/apidoc

# Default target: build HTML documentation
html:
	$(SPHINX_BUILD) -b html $(SPHINX_OPTS) $(DOCS_SOURCE) $(DOCS_BUILD)/html
	@echo "Build finished. The HTML pages are in $(DOCS_BUILD)/html."

apidoc:
	$(SPHINX_APIDOC) -o $(APIDOCS_BUILD) $(APIDOCS_SOURCE)
	@echo "API documentation generated in $(APIDOCS_BUILD)."

clean:
	rm -rf $(DOCS_BUILD)/*
	@echo "Cleaned docs build directory."

realclean: clean
	rm -rf $(APIDOCS_BUILD)/*
	@echo "Cleaned apidocs build directory."