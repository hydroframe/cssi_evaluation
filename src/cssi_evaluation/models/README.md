# Instructions for contributing a new model integration

This directory contains model-specific integrations and preprocessing utilities.

Each model integration should include functions to:

- Retrieve model outputs
- Align model outputs with observational data
- Handle coordinate transformations
- Match temporal resolution between datasets

When adding a new model integration, create a new module following the pattern:

models/<model_name>_utils.py