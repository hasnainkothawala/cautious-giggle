# Dockerfile for ML Framework
# Builds an image with the framework installed and ready to run examples using UV

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install UV (fast Python package installer) using pip
RUN pip install --no-cache-dir uv

# Copy project files
COPY . /app/

# Install dependencies and package using UV (--system flag for Docker)
RUN uv pip install --system -e .

# Verify installation
RUN python -c "import ml_framework; print('ML Framework installed successfully!')"

# Set default command
CMD ["python", "examples/conversion_prediction.py"]

