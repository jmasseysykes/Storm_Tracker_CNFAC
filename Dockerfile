# Use slim base
FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps + ca-certificates for SSL trust
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libpng-dev \
    libfreetype6-dev \
    ca-certificates \          # ← ADD THIS LINE
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "storm_tracker_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
