.PHONY: install test clean format docs-serve

install:
	pip install -e .

test:
	pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +

format:
	black src tests
	isort src tests

docs-serve:
	mdbook serve docs
