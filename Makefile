.PHONY: install lint format test test-all smoke data eval

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

data:
	uv run python scripts/prepare_mot17.py

eval:
	uv run python scripts/eval_detector.py
