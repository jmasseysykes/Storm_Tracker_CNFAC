# Use slim base for smaller image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed; matplotlib etc. sometimes want these)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app (adjust filename if not storm_tracker_app.py)
CMD ["streamlit", "run", "storm_tracker_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
