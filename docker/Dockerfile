# Use the official PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt update && \
    apt install -y gcc libfftw3-dev

# Install additional Python packages using pip


# Set the CUDA_VISIBLE_DEVICES environment variable
ENV CUDA_VISIBLE_DEVICES=1

# Copy the current directory contents into the container at /app
COPY . /workspace/hard_target

WORKDIR /workspace/hard_target

# pip install
RUN pip install --upgrade pip \
    pip install .
