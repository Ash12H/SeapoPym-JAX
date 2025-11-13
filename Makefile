.PHONY: help install install-dev test test-unit test-integration test-cov \
        format lint typecheck check clean docs docs-serve

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)SEAPOPYM-Message Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install package with core dependencies
	@echo "$(BLUE)Installing seapopym-message...$(NC)"
	uv pip install -e .

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing seapopym-message with dev dependencies...$(NC)"
	uv pip install -e ".[dev]"
	.venv/bin/pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

install-all: ## Install package with all optional dependencies
	@echo "$(BLUE)Installing seapopym-message with all dependencies...$(NC)"
	uv pip install -e ".[all]"
	.venv/bin/pre-commit install
	@echo "$(GREEN)✓ Full environment ready!$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit -m unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration -m integration -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=seapopym_message --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Lint code with ruff
	@echo "$(BLUE)Linting code...$(NC)"
	ruff check src/ tests/ --fix
	@echo "$(GREEN)✓ Code linted$(NC)"

typecheck: ## Type check with mypy
	@echo "$(BLUE)Type checking...$(NC)"
	mypy src/seapopym_message
	@echo "$(GREEN)✓ Type check passed$(NC)"

check: format lint typecheck test ## Run all quality checks (format, lint, typecheck, test)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && mkdocs build
	@echo "$(GREEN)✓ Documentation built in docs/site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://127.0.0.1:8000$(NC)"
	cd docs && mkdocs serve

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	.venv/bin/pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks passed$(NC)"

init-ray: ## Initialize Ray for distributed computing
	@echo "$(BLUE)Initializing Ray...$(NC)"
	python -c "import ray; ray.init(); print('Ray initialized successfully'); ray.shutdown()"
	@echo "$(GREEN)✓ Ray ready$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	pytest tests/ -m benchmark -v
	@echo "$(GREEN)✓ Benchmarks completed$(NC)"

notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	jupyter notebook notebooks/

version: ## Show package version
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
