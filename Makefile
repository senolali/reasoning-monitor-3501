.PHONY: install test lint run quick clean repro

install:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src python -m pytest tests/ -v

lint:
	ruff check src/ tests/

quick:
	PYTHONPATH=src python -m reasoning_monitor.cli --quick --output-dir results/quick

run:
	PYTHONPATH=src python -m reasoning_monitor.cli --output-dir results/main

repro:
	PYTHONPATH=src python -m reasoning_monitor.cli --config configs/default.yaml --output-dir results/repro

clean:
	rm -rf results/ .pytest_cache/ src/**/__pycache__/ tests/__pycache__/
