FROM python:3.13-slim

# PYTHONUNBUFFERED=1 - Ensures Python output is sent straight to the terminal without buffering.
# PYTHONDONTWRITEBYTECODE=1 - Prevents Python from writing .pyc files to disk.
# UV_SYSTEM_PYTHON=1 - Instructs the 'uv' package manager to install dependencies globally in the container's system Python, not requiring a virtual environment (venv).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /workspace

# Set the default shell to bash and enable pipefail.
# This ensures that if any command in a pipe fails (e.g., curl in 'curl | sh'), the entire RUN step fails, preventing silent errors during the build.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Update the apt package list, install essential system dependencies and clean the cache:
# --no-install-recommends - Prevents the installation of unnecessary recommended packages to keep the image lean.
# gcc - C compiler, required for building certain Python packages.
# libpq-dev - PostgreSQL header files, necessary for compiling drivers like psycopg or asyncpg.
# curl - Tool for downloading files from the web.
# git - Needed if dependencies in pyproject.toml are fetched directly from Git repositories.
# openssh-client - Provides SSH protocol support, allowing Git to authenticate with remote repositories using SSH keys.
# gnupg - Tool for managing encryption keys, useful when adding new apt repositories.
# rm -rf /var/lib/apt/lists/* - Removes the downloaded package lists, keeping the Docker image size small.

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    git \
    openssh-client \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Download and run the installation script for 'uv'
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add the directory where 'uv' was installed to the system PATH variable.
# This allows us to use the 'uv' command directly in the container's terminal without providing the full path.
ENV PATH="/root/.local/bin:$PATH"

# The '*' after uv.lock means this file will be copied if it exists, but if it doesn't, Docker won't throw an error.
COPY pyproject.toml uv.lock* ./
