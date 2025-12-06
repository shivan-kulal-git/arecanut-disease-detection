# Use Python 3.10 (stable)
FROM python:3.10-slim

# Install system dependencies needed by Pillow & numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Create work directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy entire app
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
ENV PORT=5000
EXPOSE 5000

# Start server with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1"]
