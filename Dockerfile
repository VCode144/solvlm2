FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files
COPY . /app/

# Create uploads directory
RUN mkdir -p uploads

# Expose the port FastAPI will run on
EXPOSE 7860

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 