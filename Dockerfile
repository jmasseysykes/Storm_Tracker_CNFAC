# Use slim base – trixie-slim is fine now that we drop ATLAS
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system deps only if truly needed
# build-essential + gfortran often suffice for any source compiles (rare now)
# libatlas is NOT needed — remove it
# libpng/freetype for matplotlib backends if you hit font/rendering issues later
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "storm_tracker_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
