FROM python:3.10-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# tokenizers: The Rust-backed BPE engine
# pandas: For loading CSVs
# jupyterlab: For the notebook interface
# ipywidgets: For progress bars
# jupytext: To open .py files as notebooks
RUN pip install --no-cache-dir \
    tokenizers \
    pandas \
    requests \
    jupyterlab \
    ipywidgets \
    jupytext

# Setup workspace
COPY . /app

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='musicprint'"]

