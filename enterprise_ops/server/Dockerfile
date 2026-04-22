FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire enterprise_ops package
COPY . .

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
