UV ?= uv
PYTHON ?= $(UV) run python

.PHONY: install lint format typecheck test compose-up compose-down clean deadcode

install:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run black --check .

format:
	$(UV) run black .

typecheck:
	$(UV) run mypy app

test:
	$(UV) run pytest

compose-up:
	docker compose -f docker-compose.local.yml up -d

compose-down:
	docker compose -f docker-compose.local.yml down

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache

deadcode:
	$(UV) run python scripts/find_dead_code.py
