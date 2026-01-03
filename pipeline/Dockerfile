# Use NVIDIA's optimized PyTorch container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install transformers (Hugging Face) and JupyterLab and Lightning
RUN pip install --no-cache-dir \
    transformers \
    jupyterlab \
    ipywidgets \
    pytorch-lightning \
    coremltools

WORKDIR /app

# Set python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose Jupyter port
EXPOSE 8888

CMD ["/bin/bash"]