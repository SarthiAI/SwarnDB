#!/bin/bash
# Build and publish to PyPI
set -e

# Clean previous builds
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build
python -m build

# Upload to PyPI (requires twine and PyPI credentials)
# For test PyPI: python -m twine upload --repository testpypi dist/*
# For production: python -m twine upload dist/*
echo "Built successfully. Run 'python -m twine upload dist/*' to publish."
