# Use an official Python 3.12 slim version as a parent image
FROM python:3.12-slim

# Set environment variables for Python best practices
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if needed (uncomment if faiss-cpu needs specific libs)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies from requirements.txt
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
# Using the shell form allows the $PORT variable to be substituted correctly by the shell.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
