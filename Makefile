
VENV:=qtet_venv

all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt
	./$(VENV)/bin/python3 setup.py


venv: $(VENV)/bin/activate

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

run: venv
	./$(VENV)/bin/python3 src/qtet.py -p data

.PHONY: all run venv clean
