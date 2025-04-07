# Use an official Python 3.12 slim version as a parent image
FROM python:3.12-slim

# Set environment variables for Python best practices
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if needed (uncomment if faiss-cpu needs specific libs)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# --- Explicit PyTorch CPU Installation ---
# Install torch, torchvision, torchaudio first using the official index URL for CPU.
# This often ensures better compatibility than relying solely on requirements.txt for torch.
# Using --no-cache-dir to potentially reduce image size.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# --- Install remaining dependencies ---
# Copy the requirements file
COPY requirements.txt .
# Install dependencies from requirements.txt (pip should skip torch if already installed)
# Pinned versions in requirements.txt should help here.
RUN pip install --no-cache-dir -r requirements.txt
# Install gunicorn, the production WSGI server we'll use to run Flask
RUN pip install --no-cache-dir gunicorn

# Copy the rest of the application code, data, templates, static files
COPY . .

# Make port available (Render maps this)
EXPOSE ${PORT:-5001}

# Define environment variable for the port inside the container
ENV PORT ${PORT:-5001}

# Command to run the application using Gunicorn (Shell Form)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
