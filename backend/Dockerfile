FROM python:3.11.5

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    portaudio19-dev \  
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (Good Practice)
RUN pip install --upgrade pip

# Copy requirements.txt before installing dependencies to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for the API
EXPOSE 8000

# Start Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]