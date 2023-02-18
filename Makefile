VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python3
POETRY := $(VENV_NAME)/bin/poetry

.PHONY: venv
venv:
	python3 -m venv $(VENV_NAME)
	$(PYTHON) -m ensurepip
	$(PYTHON) -m pip install poetry

.PHONY: install
install: venv
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) install
	$(POETRY) run python3 setup.py

.PHONY: clean
clean:
	rm -rf $(VENV_NAME)
