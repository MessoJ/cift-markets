# CIFT Markets - Development Makefile

.PHONY: help install dev-install setup up down restart logs clean test lint format check migrate seed

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := cift-markets

help: ## Show this help message
	@echo "CIFT Markets - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# INSTALLATION
# ============================================================================

install: ## Install production dependencies
	pip install --upgrade pip
	pip install -e .

dev-install: ## Install development dependencies
	pip install --upgrade pip
	pip install -e ".[dev,test,docs]"
	pre-commit install

setup: ## Complete initial setup (install + infrastructure + database)
	@echo "Setting up CIFT Markets..."
	@make dev-install
	@make up
	@sleep 10
	@make migrate
	@echo "✅ Setup complete! Run 'make up' to start services"

# ============================================================================
# DOCKER
# ============================================================================

up: ## Start all Docker services
	$(DOCKER_COMPOSE) up -d
	@echo "✅ Services started. Access:"
	@echo "   - QuestDB Console: http://localhost:9000"
	@echo "   - Grafana: http://localhost:3001 (admin/admin)"
	@echo "   - Prometheus: http://localhost:9090"
	@echo "   - Jaeger UI: http://localhost:16686"
	@echo "   - MLflow UI: http://localhost:5000"

down: ## Stop all Docker services
	$(DOCKER_COMPOSE) down

restart: ## Restart all Docker services
	$(DOCKER_COMPOSE) restart

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

logs-api: ## Show API logs only
	tail -f logs/cift.log

ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

# ============================================================================
# DATABASE
# ============================================================================

migrate: ## Run database migrations
	@echo "Running PostgreSQL migrations..."
	@$(DOCKER_COMPOSE) exec -T postgres psql -U cift_user -d cift_markets -f /docker-entrypoint-initdb.d/init.sql
	@echo "✅ Migrations complete"

db-shell: ## Open PostgreSQL shell
	$(DOCKER_COMPOSE) exec postgres psql -U cift_user -d cift_markets

questdb-shell: ## Open QuestDB console
	@echo "Opening QuestDB console at http://localhost:9000"
	@open http://localhost:9000 || xdg-open http://localhost:9000 || start http://localhost:9000

redis-cli: ## Open Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

# ============================================================================
# DEVELOPMENT
# ============================================================================

run-api: ## Run FastAPI development server
	uvicorn cift.api.main:app --reload --host 0.0.0.0 --port 8000

run-worker: ## Run background worker
	python -m cift.workers.main

run-stream: ## Run streaming data ingestion
	python -m cift.data.streaming.consumer

jupyter: ## Start Jupyter Lab
	jupyter lab --no-browser --port=8888

# ============================================================================
# TESTING
# ============================================================================

test: ## Run all tests
	pytest tests/ -v --cov=cift --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-watch: ## Run tests in watch mode
	ptw tests/ -- -v

coverage: ## Generate coverage report
	pytest --cov=cift --cov-report=html
	@echo "Coverage report generated at htmlcov/index.html"

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: ## Run linters (ruff + mypy)
	ruff check cift/ tests/
	mypy cift/

format: ## Format code with black and isort
	black cift/ tests/
	isort cift/ tests/

format-check: ## Check code formatting without modifying
	black --check cift/ tests/
	isort --check cift/ tests/

check: ## Run all checks (format + lint + test)
	@make format-check
	@make lint
	@make test

# ============================================================================
# CLEANING
# ============================================================================

clean: ## Clean temporary files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/

clean-all: clean down ## Clean everything including Docker volumes
	$(DOCKER_COMPOSE) down -v
	@echo "⚠️  All data volumes removed!"

# ============================================================================
# DEPLOYMENT
# ============================================================================

build: ## Build Docker images
	docker build -t $(PROJECT_NAME):latest .

push: ## Push Docker images to registry
	docker push $(PROJECT_NAME):latest

deploy-dev: ## Deploy to development environment
	@echo "Deploying to development..."
	@# Add deployment commands here

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	@# Add deployment commands here

# ============================================================================
# MONITORING
# ============================================================================

grafana: ## Open Grafana dashboard
	@echo "Opening Grafana at http://localhost:3001"
	@open http://localhost:3001 || xdg-open http://localhost:3001 || start http://localhost:3001

prometheus: ## Open Prometheus UI
	@echo "Opening Prometheus at http://localhost:9090"
	@open http://localhost:9090 || xdg-open http://localhost:9090 || start http://localhost:9090

jaeger: ## Open Jaeger tracing UI
	@echo "Opening Jaeger at http://localhost:16686"
	@open http://localhost:16686 || xdg-open http://localhost:16686 || start http://localhost:16686

mlflow: ## Open MLflow tracking UI
	@echo "Opening MLflow at http://localhost:5000"
	@open http://localhost:5000 || xdg-open http://localhost:5000 || start http://localhost:5000

# ============================================================================
# UTILITIES
# ============================================================================

shell: ## Open Python shell with app context
	ipython -i -c "from cift import settings; print('CIFT Markets shell loaded. Settings available as settings')"

env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ .env file created from template"; \
		echo "⚠️  Please update with your actual configuration"; \
	else \
		echo "⚠️  .env file already exists"; \
	fi

requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml --output-file requirements.txt

update-deps: ## Update all dependencies
	pip install --upgrade pip
	pip install --upgrade -e ".[dev,test,docs]"
