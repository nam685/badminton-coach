.PHONY: sync test lint fmt view models

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

view:
	uv run badminton-coach view

models:
	uv run badminton-coach models download
