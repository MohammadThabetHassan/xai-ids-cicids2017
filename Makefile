# XAI-IDS Makefile
# Explainable AI Intrusion Detection System

.PHONY: all install test pipeline explain clean help

# Default target
all: install pipeline

## Install dependencies
install:
	pip install -r requirements.txt

## Run the full pipeline with synthetic data
pipeline:
	python run_pipeline.py --sample-size 50000

## Run pipeline with a small dataset (for quick testing)
pipeline-small:
	python run_pipeline.py --sample-size 5000

## Run pipeline without explainability (faster)
pipeline-fast:
	python run_pipeline.py --sample-size 10000 --skip-explain

## Download real CIC-IDS-2017 dataset and run pipeline
pipeline-real:
	python run_pipeline.py --download

## Generate synthetic dataset only
generate-data:
	python -m src.data.generate_sample

## Run explainability only (requires trained models)
explain:
	python run_pipeline.py --sample-size 10000 --shap-samples 200

## Run tests
test:
	python -m pytest tests/ -v --tb=short

## Run smoke tests only
test-smoke:
	python -m pytest tests/test_smoke.py -v --tb=short -x

## Clean output artifacts
clean:
	rm -rf outputs/figures/*.png
	rm -rf outputs/models/*.pkl
	rm -rf outputs/models/*.joblib
	rm -rf outputs/logs/*.log
	rm -rf outputs/reports/*.txt
	rm -f outputs/results_metrics.csv
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__

## Show help
help:
	@echo "XAI-IDS: Explainable AI Intrusion Detection System"
	@echo ""
	@echo "Available targets:"
	@echo "  make install        Install Python dependencies"
	@echo "  make pipeline       Run full pipeline with synthetic data"
	@echo "  make pipeline-small Run with small dataset (quick test)"
	@echo "  make pipeline-fast  Run without explainability"
	@echo "  make pipeline-real  Download real dataset and run"
	@echo "  make test           Run all tests"
	@echo "  make test-smoke     Run smoke tests only"
	@echo "  make clean          Remove generated outputs"
	@echo "  make help           Show this help message"
