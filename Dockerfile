# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install git and any dependencies needed for building packages
RUN apt-get update && apt-get install -y git

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application code into the container
COPY . .

# Expose port 8000 for the FastAPI service
EXPOSE 8000

# Start uvicorn when the container launches
CMD ["uvicorn", "clip_api:app", "--host", "0.0.0.0", "--port", "8000"]
