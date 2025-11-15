FROM python:3.11-slim-bullseye

WORKDIR /app

# System dependency needed for PyCaret / LightGBM / XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
