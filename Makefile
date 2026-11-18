# Development tasks. Every Python target runs through uv, so no virtual environment
# needs to be activated first. Python tools start with python -m instead of their
# .exe launchers, which Windows application control policies can block.

UV ?= uv
IMAGE ?= digitgan:latest
DIGITS ?= 0123456789
VARIATIONS ?= 4
MODEL_URL ?= https://github.com/Samiicodes/handwritten-digit-gan/releases/download/v0.1.0/cgan_seed42_generator.pt

.PHONY: help install download-model lint format test train train-cgan evaluate docs generate serve docker-build docker-run

help:
	@echo install         Install dependencies and the pre-commit hooks
	@echo download-model  Download the trained conditional GAN that the API serves
	@echo lint            Run ruff and mypy
	@echo format          Apply ruff fixes and formatting
	@echo test            Run the test suite with coverage
	@echo train           Train the classifier and every GAN for every seed
	@echo train-cgan      Train only the conditional GAN that the API serves, one seed
	@echo evaluate        Score every run and write reports/evaluation_report.md
	@echo docs            Evaluate and copy the report and figures into docs/
	@echo generate        Write DIGITS in generated handwriting, VARIATIONS times
	@echo serve           Start the API on http://127.0.0.1:8000
	@echo docker-build    Build the Docker image
	@echo docker-run      Serve the API from the image, using the generator in runs/

install:
	$(UV) sync --frozen
	$(UV) run python -m pre_commit install

download-model:
	curl --fail --location --create-dirs --output runs/cgan_seed42/generator.pt $(MODEL_URL)

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run python -m mypy

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run python -m pytest --cov --cov-report=term-missing

train:
	$(UV) run python scripts/train_models.py

train-cgan:
	$(UV) run python scripts/train_models.py --models cgan --seeds 42

evaluate:
	$(UV) run python scripts/evaluate_models.py

docs:
	$(UV) run python scripts/evaluate_models.py --publish_docs

generate:
	$(UV) run python scripts/generate_digits.py --digits $(DIGITS) --variations $(VARIATIONS)

serve:
	$(UV) run python scripts/run_api.py

docker-build:
	docker build --tag $(IMAGE) .

docker-run:
	docker run --rm --publish 8000:8000 --volume "$(CURDIR)/runs:/app/runs:ro" $(IMAGE)
