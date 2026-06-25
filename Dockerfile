# Use slim image to reduce memory footprint (important for Render 512MB plans)
FROM python:3.11-slim

WORKDIR /app

# Install build deps for matplotlib + CA certs for NRCS/Supabase SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libpng-dev \
    libfreetype6-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (no cache to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: reduce matplotlib memory if possible
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

# Copy the rest of the application
COPY . .

EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Storm_Tracker.py", "--server.port=8501", "--server.address=0.0.0.0"]
