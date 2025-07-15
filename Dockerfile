# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Expose port for the FastAPI app
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# # Start from Python base image
# FROM python:3.11-slim
#
# # Set working directory inside the container
# WORKDIR /app
#
# # Copy pre-downloaded wheels and install requirements
# COPY wheels /wheels
# COPY requirements.txt .
# RUN pip install --no-cache-dir --find-links=/wheels -r requirements.txt
#
# # Copy application source code
# COPY app /app
#
# # Run your app (adjust as needed)
# # CMD ["python", "app.main.py"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# # Use official Python image
# FROM python:3.11
#
# # Set working directory inside container
# WORKDIR /app
#
# # Copy the requirements file and install dependencies
# # COPY requirements-prod.txt ./requirements.txt
# # RUN pip install --no-cache-dir -r requirements.txt
#
# COPY requirements.txt .
#
# RUN pip install --no-cache-dir -r requirements.txt
#
# # Copy the entire project into the container
# COPY . .
#
# # Expose port for the FastAPI app
# EXPOSE 8000
#
# # Command to run the app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
