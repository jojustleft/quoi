.PHONY: venv
venv: # Create virtual environment and install all requirements
	uv venv --clear
	uv sync

.PHONY: test
test: # Run test suite
	uv run pytest --verbose

.PHONY: jupyter
jupyter: # Start Jupyter Lab instance
	uv run jupyter lab

.PHONY: ruff
ruff:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: clean
clean: # Remove all temporary files / folders and the virtual environment
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .venv
	rm -rf build
	rm -rf quoi.egg-info
