# =============================================================================
# YADEM AI Engine — Makefile
# =============================================================================

.PHONY: install train serve test lint clean docker-build docker-up

# Install all dependencies
install:
	pip install -r requirements.txt

# Generate synthetic data and train all models
train:
	python train.py

# Start the FastAPI server (development)
serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Lint code
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Format code
format:
	ruff format src/ tests/

# Clean artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# Build Docker image
docker-build:
	docker build -f docker/Dockerfile.api -t yadem-api:latest .

# Start full stack with Docker Compose
docker-up:
	docker-compose -f docker/docker-compose.yml up -d

# Stop Docker stack
docker-down:
	docker-compose -f docker/docker-compose.yml down

# Full pipeline: install, train, test, serve
pipeline: install train test serve
