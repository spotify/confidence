.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts


clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

format: ## format code with black
	black spotify_confidence tests --line-length 119

lint: ## check style with flake8
	flake8 spotify_confidence tests

test: ## run tests quickly with the default Python
	python3 -m pytest

coverage: ## check code coverage quickly with the default Python
	coverage run --source spotify_confidence -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

# docs: ## generate Sphinx HTML documentation, including API docs
# 	rm -f docs/confidence.rst
# 	rm -f docs/modules.rst
# 	sphinx-apidoc -o docs/ confidence
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	$(BROWSER) docs/_build/html/index.html

# servedocs: docs ## compile the docs watching for changes
# 	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release-test: clean ## package and upload a release
	python3 -m build
	python3 -m twine upload --repository testpypi dist/*

release-prod: clean ## package and upload a release
	python3 -m build
	python3 -m twine upload dist/*

dist: clean ## builds source and wheel packagels
	python3 -m build
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install -e .

install-test: clean
	pip3 install --index-url https://test.pypi.org/simple/ spotify-confidence

install-prod: clean
	pip3 install spotify-confidence

