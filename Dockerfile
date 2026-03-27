# Use full Python image (includes complete CA certificates - fixes NRCS SSL error)
FROM python:3.11

WORKDIR /app

# Install only the build dependencies we actually need for matplotlib/scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "storm_tracker_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
