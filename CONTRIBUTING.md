# Contributing

Contributions are welcome when they preserve the repository's public safety
boundary.

## Development

1. Create a Python 3.11 or newer virtual environment.
2. Install the locked dependencies with:
   python -m pip install --require-hashes -r requirements-dev.lock
3. Install the project with: python -m pip install -e . --no-deps
4. Run: python -m ruff check .
5. Run: python -m mypy src
6. Run: python -m pytest --cov

## Public-data boundary

- Use only synthetic fixtures or data with a documented redistribution license.
- Do not add exchange credentials, account identifiers, deployment details,
  private endpoints, environment files, or real trade records.
- Do not add strategy parameters or research artifacts copied from a private
  system.
- New methodology must cite a primary source and include deterministic tests.

By contributing, you certify that you have the right to submit the contribution
under the repository's MIT License.
