# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade CMake to version 3.5+ (required for dlib)
RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get remove -y cmake && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main' && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install cmake first
RUN pip install --upgrade pip setuptools wheel && \
    pip install cmake>=3.5

# Install dlib separately (requires compilation)
RUN pip install dlib==19.24.2 || \
    (pip install cmake && pip install dlib==19.24.2)

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=fixed_integrated_attendance_system.py

# Run the application
CMD ["python", "fixed_integrated_attendance_system.py"]

