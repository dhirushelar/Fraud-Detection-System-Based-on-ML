# ==============================================================================
# Fraud Detection System (FDS) - Automation Makefile
# Author: Alpha
# ==============================================================================

.PHONY: help setup data train seed run test lint docker-up docker-down clean

PYTHON ?= python
PIP ?= pip

help:
	@echo "Fraud Detection System (FDS) - Automation Commands"
	@echo "--------------------------------------------------"
	@echo "make setup       : Install production and development dependencies"
	@echo "make data        : Generate synthetic dataset (5,000 transactions)"
	@echo "make train       : Fit pipeline, train Isolation Forest & export artifacts"
	@echo "make seed        : Initialize database tables, seed RBAC accounts & transactions"
	@echo "make run         : Start local Flask development server (port 5000)"
	@echo "make test        : Execute entire pytest suite (29 tests)"
	@echo "make lint        : Run code formatting and lint checks via Ruff"
	@echo "make docker-up   : Spin up Docker Compose stack (App + Postgres + MLflow)"
	@echo "make docker-down : Tear down Docker Compose containers and networks"
	@echo "make clean       : Remove bytecode, test caches, and temporary artifacts"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "[+] Initialized .env from template"; fi

data:
	$(PYTHON) scripts/generate_data.py --count 5000 --fraud-ratio 0.005

train:
	$(PYTHON) scripts/train_model.py

seed:
	$(PYTHON) scripts/seed_db.py

run:
	$(PYTHON) wsgi.py

test:
	$(PYTHON) run_tests.py -v

lint:
	ruff check .
	ruff format --check .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
