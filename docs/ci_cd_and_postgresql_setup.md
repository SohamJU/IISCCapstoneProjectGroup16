# CI, CD, and PostgreSQL Setup Overview

## Purpose

This guide explains how the repository is tested, deployed, and connected to PostgreSQL. It is intended for both developers and non-technical stakeholders who want to understand the operational model behind the project.

## What this repository does operationally

The project uses three connected layers:

1. CI to validate code changes before they are merged.
2. CD to publish the Gradio app to a demo environment.
3. PostgreSQL as the structured data store for customer, order, return, and product information.

Together, these layers support reliable releases and a more dependable customer support experience.

## CI: Continuous Integration

The CI workflow lives in [.github/workflows/ci.yml](../.github/workflows/ci.yml). It is triggered on pushes to the main branch and on pull requests targeting main.

From a developer perspective, the workflow does the following:

- Checks out the repository
- Sets up Python 3.12
- Creates a virtual environment with uv
- Installs project dependencies with uv sync
- Runs Ruff for linting
- Runs MyPy for type checking
- Runs Pytest for the test suite with coverage

In practical terms, this means that a contributor can expect automated quality checks whenever a change is proposed or merged.

## CD: Continuous Delivery and Deployment

The deployment workflow is defined in [.github/workflows/cd.yml](../.github/workflows/cd.yml). It is designed to publish the app to a Hugging Face Space so the project can be demonstrated or reviewed easily.

The workflow performs the following steps:

- Checks out the repository
- Configures Git for the deployment environment
- Uses the GitHub secrets HF_TOKEN and HF_SPACE to authenticate
- Copies the app and source files into a deployment folder
- Writes a minimal Hugging Face Space README and requirements file
- Pushes the updated app to the target Space

For developers, the important point is that deployment is automated and driven by repository content plus secrets, rather than manual copy-and-paste steps.

## PostgreSQL setup

The application uses PostgreSQL as its structured data layer. The connection settings are defined in [src/config/data.py](../src/config/data.py), and the project expects the following environment variables to be present:

- POSTGRESQL_HOST
- POSTGRESQL_PORT
- POSTGRESQL_USER
- POSTGRESQL_DB
- POSTGRESQL_AIVEN_PASSWORD

The code builds a SQLAlchemy connection string using these values, with SSL enabled. The repository is currently wired to use an Aiven-hosted PostgreSQL instance.

### What data is expected in PostgreSQL

The configuration points to a set of relational tables that include:

- customers
- orders
- order_items
- returns
- customer_queries
- product_catalog
- reviews

These tables are used by the agent system to answer questions and provide support-related responses based on real or generated data.

## Developer notes

If you are working on this repo, the practical workflow is:

1. Make changes locally.
2. Run the same checks that CI would run: linting, type checking, and tests.
3. Push your branch and open a pull request.
4. If deployment is needed, the CD workflow will publish the app to the Hugging Face Space using the configured secrets.
5. Ensure that your local environment contains the required PostgreSQL variables before running data or agent workflows.

## Why this matters for the business

From a business perspective, this setup helps the team:

- Release updates more consistently
- Reduce the risk of broken or incomplete changes
- Make demos easier to run and share
- Keep the application connected to trustworthy operational data

## Recommended next steps

As the project grows, the team should consider:

- Adding a deployment health check after publishing
- Documenting backup and recovery procedures for PostgreSQL
- Creating separate staging and production environments
- Clarifying ownership for release and support operations

## Summary

CI keeps the codebase healthy, CD makes releases easier to manage, and PostgreSQL provides the structured data foundation for the application. The repository is already wired to support these workflows in a practical, developer-friendly way.
