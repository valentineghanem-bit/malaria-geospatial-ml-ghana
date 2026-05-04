FROM python:3.11-slim

LABEL maintainer="valentineghanem@gmail.com"
LABEL description="Ghana Malaria 260-District Geospatial & ML Analysis"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgdal-dev libgeos-dev libproj-dev \
    gdal-bin git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8050

CMD ["python", "scripts/figures/generate_figures.py"]
