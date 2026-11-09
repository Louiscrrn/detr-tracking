.PHONY: install lint format test test-all smoke

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-all:
	uv run pytest -m ""

smoke:
	uv run python scripts/smoke_detect.py
