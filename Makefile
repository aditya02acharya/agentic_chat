.PHONY: install dev test lint format clean run

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest tests/ -v --cov=src/agentic_chatbot --cov-report=term-missing

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Run integration tests only
test-integration:
	pytest tests/integration/ -v

# Run linting
lint:
	ruff check src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	ruff check --fix src/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run development server
run:
	uvicorn agentic_chatbot.main:app --reload --host 0.0.0.0 --port 8000

# Run production server
run-prod:
	uvicorn agentic_chatbot.main:app --host 0.0.0.0 --port 8000 --workers 4

# Type checking
typecheck:
	mypy src/

# Generate requirements.txt from pyproject.toml
requirements:
	pip-compile pyproject.toml -o requirements.txt
