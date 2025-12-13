.PHONY: all install build-kernels clean test docs

# Default target
all: install

# Install the package in development mode
install:
	pip install -e .

# Build custom CUDA kernels
build-kernels:
	python setup_kernels.py build_ext --inplace

# Alternative build method using the shell script
build-kernels-sh:
	./build_kernels.sh

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.so" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Run tests (when implemented)
test:
	python -m pytest tests/

# Format code
format:
	black src/

# Lint code
lint:
	flake8 src/

# Install development dependencies
dev:
	pip install -e .[dev]

# Build documentation (placeholder)
docs:
	@echo "Documentation generation would happen here"
	@echo "For now, please refer to README.md and guide.md"
