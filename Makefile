    .PHONY: install format lint test run

    install:
	pip install -e .[dev]

    format:
	black .

    lint:
	ruff check .

    test:
	pytest -q

    run:
	convobot
