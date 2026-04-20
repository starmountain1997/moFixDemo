FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y git && \
    pip install uv

# Clone the repository
RUN git clone https://gitcode.com/Ascend/msmodeling.git

# Copy pyproject.toml
COPY ms_pyproject.toml msmodeling/pyproject.toml

# Install Python dependencies
RUN pip install -r msmodeling/requirements.txt

# Set working directory to the project
WORKDIR /app/msmodeling

# Sync dependencies
RUN uv sync

CMD ["/bin/bash"]
