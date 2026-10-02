.PHONY: check lint format test build clean install help

PYTHON        := python3
SRC_DIR       := src
TEST_DIR      := tests
BUILD_DIR     := build
FUNCTION_ZIP  := function.zip
COV_MIN       := 90

help:
	@echo "Targets:"
	@echo "  make install  - install dev dependencies"
	@echo "  make check    - format, lint, and test (local dev flow)"
	@echo "  make build    - package src/ into function.zip for deployment"
	@echo "  make clean    - remove build artifacts and caches"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt boto3

## ---- Flow 1: format, lint, test ----
check: format lint test

format:
	ruff format $(SRC_DIR) $(TEST_DIR)

lint:
	ruff check $(SRC_DIR) $(TEST_DIR)

test:
	pytest --cov=$(SRC_DIR) --cov-report=term-missing --cov-fail-under=$(COV_MIN)

## ---- Flow 2: build deployment package ----
build: clean
	mkdir -p $(BUILD_DIR)
	cp $(SRC_DIR)/handler.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip -r ../$(FUNCTION_ZIP) .
	@echo "Built $(FUNCTION_ZIP) ready for deployment"

clean:
	rm -rf $(BUILD_DIR) $(FUNCTION_ZIP) .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
