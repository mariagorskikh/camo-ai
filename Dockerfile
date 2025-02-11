FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=8000

# Expose port (this is just documentation, Railway will override it)
EXPOSE ${PORT}

# Run the application with the PORT environment variable
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app", "--workers", "1", "--threads", "2", "--timeout", "30"]
