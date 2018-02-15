.PHONY: default-target
default-target: build

###############################################################################
### User-settable variables

# Commands to run in the build process.
PYTHON = python
SPHINX_BUILD = sphinx-build
SPHINX_FLAGS =

# Options for above commands.
SPHINXOPTS =
PYTHONOPTS =
SETUPPYOPTS =

###############################################################################
### Targets

# build: Build trcrpm python package.
.PHONY: build
build: setup.py
	$(PYTHON) $(PYTHONOPTS) setup.py $(SETUPPYOPTS) build

# doc: Build the trcrpm documentation.
.PHONY: doc
doc: pythenv.sh build
	rm -rf build/doc.tmp && \
	./pythenv.sh $(SPHINX_BUILD) $(SPHINX_FLAGS) \
		-b html doc/ build/doc.tmp && \
	rm -rf build/doc && \
	mv -f build/doc.tmp build/doc

# check: (Build trcrpm and) run the tests.
.PHONY: check
check: check.sh
	./check.sh

# clean: Remove build products.
.PHONY: clean
clean:
	-rm -rf build
